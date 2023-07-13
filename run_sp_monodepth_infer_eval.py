import os, argparse, tqdm

import torch
import numpy as np
import healpy as hp
import open3d as o3d
import cv2

from config import *
from libs.logger import *
from libs.model import SwinSphDecoderNet, ResnetSphDecoderNet, EffnetSphDecoderNet
from libs.dataset import *
from libs.util_helper import to_numpy
from libs.metrics import AverageMeter, Evaluator
from libs.metrics_sphere import SphericalEvaluator, Sample


def render_depth_map(hp_data, image_to_sp):
    return hp_data[:, :, :, image_to_sp].squeeze(2)


def compute_hp_info(nside=128, img_size=(512, 1024)):
    hp_info = {}
    h, w = img_size[0], img_size[1]
    pixel_num_sp = hp.nside2npix(nside)
    pixel_idx = np.arange(pixel_num_sp)
    sp_ll = hp.pix2ang(nside, pixel_idx, nest=True, lonlat=True)

    x, y = sp_ll[0] / 360.0 * w, (sp_ll[1] + 90.0) / 180.0 * h
    x, y = x.reshape(1, -1), y.reshape(1, -1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    sp_to_image = np.concatenate([x0, y0, x1, y1, wa, wb, wc, wd], axis=0)

    sp_xyz = hp.pix2vec(nside, pixel_idx, nest=True)
    sp_xyz = np.stack([sp_xyz[0], sp_xyz[1], sp_xyz[2]], axis=1)

    # consider half pixels
    theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    theta = np.pi - np.repeat(theta, w, axis=1)
    theta = theta.flatten()
    phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w
    phi = np.repeat(phi, h, axis=0)
    phi = phi.flatten()
    # see healpy doc:
    # https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.ang2pix.html#healpy.pixelfunc.ang2pix
    # `lonlat=False` to use colattitude representation
    image_to_sp = hp.pixelfunc.ang2pix(nside, theta, phi, nest=True, lonlat=False)
    hp_info["hp_dir"] = sp_xyz
    hp_info["hp_pix_num"] = pixel_num_sp
    hp_info["hp_to_image_map"] = sp_to_image
    hp_info["image_to_sp_map"] = torch.from_numpy(image_to_sp.reshape(h, w))
    return hp_info


def save_pcd(xyz, rgb, out_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)


def compute_error_heatmap(depth_map, gt_depth, background, blue_mask_weight=0.1,
                          heat_map_weight=1.0, min_depth=0.0001, max_depth=10.0):
    assert depth_map.shape == gt_depth.shape
    h, w = depth_map.shape
    mask = (gt_depth < min_depth) | (gt_depth > max_depth)
    error_map = np.fabs(depth_map - gt_depth) / (gt_depth + 1e-8)
    error_map[mask] = 0.0

    # bg_img_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
    error_map_norm = ((error_map - error_map.min()) / (error_map.max() - error_map.min()) * 255).astype(np.uint8)
    #blue_mask = cv2.rectangle(background.copy(), (0, 0), (w, h), (0, 0, 255)).astype(np.uint8)
    #base = cv2.addWeighted(blue_mask, blue_mask_weight, background, 1.0 - blue_mask_weight, 0)

    #hotmap_blue = np.array(cv2.applyColorMap(error_map_norm, cv2.COLORMAP_JET))
    #hotmap_red = cv2.cvtColor(hotmap_blue, cv2.COLOR_RGB2BGR)
    #intensity_hotmap_img = cv2.addWeighted(hotmap_blue, heat_map_weight, base, 1 - heat_map_weight, 0)

    #return intensity_hotmap_img
    return error_map_norm


def main():
    parser = argparse.ArgumentParser(description="360 degree panorama depth estimation inference and evaluation")
    parser.add_argument("-d", "--dataset_root_dir", default="", type=str, help="The dataset root dir")
    parser.add_argument("-i", "--input_list", default="", type=str, help="The input file list")
    parser.add_argument("-o", "--output_dir", default="", type=str, help="Output directory")
    parser.add_argument("-m", "--model_path", type=str, default="", help="model path")
    parser.add_argument("--nside", type=int, default="128", help="The output healpix nside")
    parser.add_argument("-c", "--cfg", type=str, default="", help="The configuration file for inferring or evaluation")
    parser.add_argument("--max_depth_meters", type=float, default=10.0, help="maximum depth meters range")
    parser.add_argument("--min_depth_meters", type=float, default=0.1, help="minimum depth meters range")
    parser.add_argument('--eval_depth_map', action='store_true', default=False,
                        help="Do evaluation on eqr depth map with gt depth if true")
    parser.add_argument('--eval_hp', action='store_true', default=False,
                        help="Do evaluation on healpix sphere with gt depth if true")
    parser.add_argument('--eval_err_heatmap', action='store_true', default=False,
                        help="Do evaluation on error distribution")
    parser.add_argument('--save_visualization', action='store_true', default=False,
                        help="Save visualization results: point clouds, output depth, ...")
    parser.add_argument("--depth_scale", type=float, default=4000.0, help="depth scale applied to predicted depth map")
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")

    args = parser.parse_args()
    eval_hp = args.eval_hp
    eval_depth_map = args.eval_depth_map
    eval_mode = eval_depth_map or eval_hp
    save_vis = args.save_visualization
    eval_err_heatmap = args.eval_err_heatmap
    assert args.cfg != "", "Configuration file should be specified"
    cfg = get_config(args.cfg)

    if args.output_dir != "":
        cfg.defrost()
        cfg.OUT_ROOT_DIR = args.output_dir
        cfg.freeze()
    assert cfg.OUT_ROOT_DIR != '', "Output directory must be specified!"
    os.makedirs(cfg.OUT_ROOT_DIR, exist_ok=True)
    vis_cloud_dir = os.path.join(cfg.OUT_ROOT_DIR, "vis_clouds")
    os.makedirs(vis_cloud_dir, exist_ok=True)
    vis_depth_dir = os.path.join(cfg.OUT_ROOT_DIR, "vis_depth")
    os.makedirs(vis_depth_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.defrost()
        cfg.TESTING.RANK = int(os.environ["RANK"])
        cfg.TESTING.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        cfg.freeze()
    assert cfg.TESTING.WORLD_SIZE == torch.cuda.device_count()
    print(f"=>RANK and WORLD_SIZE in environ: {cfg.TESTING.RANK}/{cfg.TESTING.WORLD_SIZE}")

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend=cfg.TESTING.BACKEND, init_method=cfg.TESTING.INIT_METHOD,
                                         world_size=cfg.TESTING.WORLD_SIZE, rank=cfg.TESTING.RANK)
    logger = create_logger(output_dir=cfg.OUT_ROOT_DIR, dist_rank=dist.get_rank(),
                           file_name_prefix="log_eval" if eval_mode else "log_infer")
    logger.info("Init logging system!")

    # master process to dump used configurations
    if dist.get_rank() == 0:
        config_path = os.path.join(cfg.OUT_ROOT_DIR, "config.json")
        with open(config_path, "w") as f:
            f.write(cfg.dump())
        logger.info(f"Exporting training configuration as json: {config_path}")

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    encoder_model_dict = {"swin": SwinSphDecoderNet,
                          "resNet": ResnetSphDecoderNet,
                          "effnet": EffnetSphDecoderNet}
    model_type = encoder_model_dict[cfg.BACKBONE.TYPE]
    model = model_type(cfg, pretrained=False)
    model.to(device)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    if checkpoint is None:
        logger.info(f"Load model file failed: {args.model_path}")
        exit(-1)
    assert 'model' in checkpoint, 'Model not exists in checkpoint file'
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    logger.info(msg)
    del checkpoint
    logger.info(f"Load model file successfully: {args.model_path}")
    torch.cuda.empty_cache()
    model.eval()

    datasets_dict = {"Matterport3D": M3DDatasetAug,
                     "Stanford3D": StanfordDatasetAug,
                     "PanoSUNCG3D": PanoSUNCGDatasetAug,
                     "3D60": ThreeD60DatasetAug,
                     "Pano3D": Pano3DAugDataset}

    dataset = datasets_dict[cfg.DATA.DATASET_NAME]
    dataset = dataset(args.dataset_root_dir, args.input_list, cfg, do_augmentation=False,
                      keep_raw_rgb=True if save_vis else False, keep_raw_gt_depth=True if eval_depth_map else False)
    data_loader = init_loader(cfg, dataset, is_train=False)

    num_samples = len(dataset)
    num_steps = num_samples // cfg.TESTING.BATCH_SIZE + 1
    logger.info(f"Num. of samples:{num_samples}, Num. of steps: {num_steps}")

    hp_info = compute_hp_info(args.nside, (cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description("Evaluation" if eval_mode else "Inferring")
    htim = hp_info["hp_to_image_map"]
    coord_map = htim[0:4, :].astype(np.int32)
    weight_map = htim[4:8, :].astype(np.float)
    x0, y0, x1, y1 = coord_map[0, :], coord_map[1, :], coord_map[2, :], coord_map[3, :]
    wa, wb, wc, wd = weight_map[0, :], weight_map[1, :], weight_map[2, :], weight_map[3, :]
    wa, wb, wc, wd = wa.reshape(-1, 1), wb.reshape(-1, 1), wc.reshape(-1, 1), wd.reshape(-1, 1)
    if eval_hp:
        hp_evaluator = Evaluator(median_align=cfg.TESTING.MEDIAN_ALIGN)
    if eval_depth_map:
        if cfg.DATA.DATASET_NAME == "Pano3D":
            depth_map_evaluator = SphericalEvaluator(median_align=cfg.TESTING.MEDIAN_ALIGN)
            vertices = Sample(6, 1024, 'nearest', 'mesh')
        else:
            depth_map_evaluator = Evaluator(cfg.TESTING.MEDIAN_ALIGN)

    with torch.no_grad():
        for bid, inputs in enumerate(pbar):
            rgb = inputs["rgb"].to(device)
            if cfg.DATA.DATASET_NAME == "Pano3D":
                weight = inputs["eval_weight"].to(device)
            pred_depths_hp = model(rgb)
            depth_maps = render_depth_map(pred_depths_hp, hp_info["image_to_sp_map"])
            if eval_hp:
                gt_depth = inputs["gt_depth"].to(device)
                gt_mask = inputs["mask"].to(device)
                hp_evaluator.compute_eval_metrics(gt_depth, pred_depths_hp, gt_mask, max_depth=cfg.MODEL.MAX_DEPTH)
            if eval_depth_map:
                gt_depth = inputs["raw_gt_depth"].to(device)
                gt_mask = inputs["raw_gt_depth_mask"].to(device)
                if cfg.DATA.DATASET_NAME == "Pano3D":
                    gt_depth_sphere = vertices.forward(gt_depth.detach().cpu())
                    depth_maps_sphere = vertices.forward(depth_maps.detach().cpu())
                    depth_map_evaluator.compute_eval_metrics(gt_depth_sphere, depth_maps_sphere,
                                                             max_depth=cfg.MODEL.MAX_DEPTH, weight=weight)
                else:
                    depth_map_evaluator.compute_eval_metrics(gt_depth, depth_maps, gt_mask)
            if save_vis:
                # 1. save colored point cloud
                depths_hp = to_numpy(pred_depths_hp).transpose(0, 2, 3, 1)
                depth_maps = to_numpy(depth_maps).transpose(0, 2, 3, 1)
                cond = (depths_hp < args.max_depth_meters) & (depths_hp > args.min_depth_meters)
                depths_hp = np.where(cond, depths_hp, 0)
                batch_size = depths_hp.shape[0]
                raw_rgb = to_numpy(inputs["raw_rgb"])
                names = inputs["name"]
                for i in range(batch_size):
                    output_ply = os.path.join(vis_cloud_dir, names[i] + ".ply")
                    depth = depths_hp[i, :, :, :].squeeze(0).repeat(3, axis=-1)
                    points = depth * hp_info["hp_dir"]
                    raw_color = raw_rgb[i, :, :, :]
                    color = wa * raw_color[y0, x0, :] + wb * raw_color[y1, x0, :] + \
                            wc * raw_color[y0, x1, :] + wd * raw_color[y1, x1, :]
                    color = np.clip(color, 0.0, 255.0) / 255.0
                    save_pcd(points, color, output_ply)
                    depth_map = depth_maps[i, :, :, :].squeeze(-1)
                    if eval_err_heatmap:
                        gt_depths = to_numpy(inputs["raw_gt_depth"])
                        gt_depth = gt_depths[i, :, :].squeeze(0)
                        output_heatmap_path = os.path.join(vis_cloud_dir, names[i] + "_heat.jpg")
                        heat_map = compute_error_heatmap(depth_map, gt_depth, raw_color)
                        cv2.imwrite(output_heatmap_path, heat_map)
                    depth_map = (depth_map * args.depth_scale).astype(np.uint16)
                    depth_image_path = os.path.join(vis_depth_dir, names[i] + ".png")
                    cv2.imwrite(depth_image_path, depth_map)
        logger.info("Run all inferring/evaluation done!")
    if eval_hp:
        hp_evaluator.print()
    if eval_depth_map:
        depth_map_evaluator.print()


if __name__ == "__main__":
    main()
