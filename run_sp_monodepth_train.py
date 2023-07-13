"""
The script is used to train a depth estimation neural model for panoramic(equi-rectangular) images.
We implement this training process based on Swin(https://github.com/microsoft/Swin-Transformer)
"""

import argparse, time, tqdm, datetime
import numpy as np

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from libs.logger import *

from libs.metrics import *
from libs.dataset import *
from libs.optimizer import init_optimizer, init_scheduler
from libs.util_helper import *
from libs.loss import BerhuLoss, RMSELog
from libs.model import SwinSphDecoderNet, ResnetSphDecoderNet, EffnetSphDecoderNet

try:
   from apex import amp
except ImportError:
   amp = None

from config import get_config


def adaptive_train_params(cfg):
    # linear scale the learning rate according to total batch size, steal it from "Swintransfomer"
    linear_scaled_lr = cfg.TRAIN.BASE_LR * cfg.TRAIN.BATCH_SIZE * dist.get_world_size()
    linear_scaled_warmup_lr = cfg.TRAIN.WARMUP_LR * cfg.TRAIN.BATCH_SIZE * dist.get_world_size()
    linear_scaled_min_lr = cfg.TRAIN.MIN_LR * cfg.TRAIN.BATCH_SIZE * dist.get_world_size()
    # gradient accumulation also need to scale the learning rate
    if cfg.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * cfg.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * cfg.TRAIN.ACCUMULATION_STEPS

    # update learning rate adaptive to computation configuration
    cfg.defrost()
    cfg.TRAIN.BASE_LR = linear_scaled_lr
    cfg.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    cfg.TRAIN.MIN_LR = linear_scaled_min_lr
    cfg.freeze()

    return cfg


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, device, writer_train):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    loss_type_dict = {"RMSLE": RMSELog,
                      "BerHu": BerhuLoss}
    compute_loss = loss_type_dict[config.TRAIN.LOSS_TYPE]()

    for idx, inputs in enumerate(data_loader):
        rgb, gt_depth, mask = inputs["rgb"], inputs["gt_depth"], inputs["mask"]

        rgb = rgb.to(device, non_blocking=True)
        gt_depth = gt_depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred_depth = model(rgb)

        loss = compute_loss(gt_depth, pred_depth, mask)

        step = epoch * num_steps + idx
        if cfg.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / cfg.TRAIN.ACCUMULATION_STEPS
            if cfg.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                # clip grad to ensure stable training
                if cfg.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if cfg.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % cfg.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(step)
        else:
            optimizer.zero_grad()
            if cfg.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if cfg.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(step)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), gt_depth.size(0)) # consider batch size
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  '
                f'mem {memory_used:.0f}MB')
            writer_train.add_scalar("train_loss", loss_meter.avg, step)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(cfg, data_loader, model, device):
    model.eval()
    evaluator = Evaluator()
    pbar = tqdm.tqdm(data_loader)
    pbar.set_description("Validating")
    batch_eval_time = AverageMeter()
    end = time.time()
    loss_type_dict = {"RMSLE": RMSELog,
                      "BerHu": BerhuLoss}
    compute_loss = loss_type_dict[cfg.TRAIN.LOSS_TYPE]()
    loss_meter = AverageMeter()

    for bidx, inputs in enumerate(data_loader):
        rgb, gt_depth, mask = inputs["rgb"], inputs["gt_depth"], inputs["mask"]
        rgb = rgb.to(device, non_blocking=True)
        gt_depth = gt_depth.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred_depth = model(rgb)

        evaluator.compute_eval_metrics(gt_depth, pred_depth, mask)
        loss = compute_loss(gt_depth, pred_depth, mask)
        loss_meter.update(loss.item(), gt_depth.size(0))  # consider batch size
        # measure elapsed time
        batch_eval_time.update(time.time() - end)
        end = time.time()

        if bidx % cfg.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Validation: [{bidx}/{len(data_loader)}]\t'
                        f'Val/eval Time {batch_eval_time.val:.3f} ({batch_eval_time.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})')
            evaluator.print()
    eval_metric = {}
    for i, key in enumerate(evaluator.metrics.keys()):
        eval_metric[key] = np.array(reduce_tensor(evaluator.metrics[key].avg).cpu())
    eval_metric["val/loss"] = np.array(loss_meter.avg)
    return eval_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root_dir", default="", help="dataset root dir")
    parser.add_argument("-i", "--train_data_list", default="", help="input training data file list")
    parser.add_argument("-v", "--valid_data_list", default="", help="input training data file list")

    parser.add_argument("-o", "--output_dir", default="", help="output directory for trainning process")
    parser.add_argument("-c", "--train_cfg", default="", help="yaml file for training configuration")
    parser.add_argument("-p", "--pretrained", default="", help="the offline pretrained weights for swin")
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")
    args = parser.parse_args()
    assert args.train_cfg != "", "Training configuration file should be specified!"
    # get configurations from cfg file that can be replaced by cmd-line options
    cfg = get_config(args.train_cfg)

    # update configs by cmd-line options
    cfg.defrost()
    cfg.OUT_ROOT_DIR = args.output_dir
    pretrained = False
    if args.pretrained != '':
        cfg.TRAIN.PRETRAINED_MODEL = args.pretrained
        pretrained = True
    cfg.freeze()
    assert cfg.OUT_ROOT_DIR != '', "Output directory must be specified!"

    os.makedirs(cfg.OUT_ROOT_DIR, exist_ok=True)
    # logging and train/val events output directory
    out_logging = os.path.join(cfg.OUT_ROOT_DIR, "logging")
    out_logging_train = os.path.join(out_logging, "train")
    out_logging_val = os.path.join(out_logging, "val")
    os.makedirs(out_logging_train, exist_ok=True)
    os.makedirs(out_logging_val, exist_ok=True)

    out_models = os.path.join(cfg.OUT_ROOT_DIR, "models")
    os.makedirs(out_models, exist_ok=True)

    if cfg.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "Use amp optimization, check apex package installation!"

    # environment variables or command line options have higher priority
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.defrost()
        cfg.TRAIN.RANK = int(os.environ["RANK"])
        cfg.TRAIN.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        cfg.freeze()

    print(f"=> RANK and WORLD_SIZE in environ: {cfg.TRAIN.RANK}/{cfg.TRAIN.WORLD_SIZE}")

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend=cfg.TRAIN.BACKEND, init_method=cfg.TRAIN.INIT_METHOD,
                                         world_size=cfg.TRAIN.WORLD_SIZE, rank=cfg.TRAIN.RANK)
    torch.distributed.barrier()

    seed = cfg.SEED + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', args.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    logger = create_logger(output_dir=out_logging, dist_rank=dist.get_rank(), file_name_prefix="log_train_rank")
    logger.info("Init logging system!")
    # adjust training hyper-parameters
    cfg = adaptive_train_params(cfg)
    # master process to dump used configurations
    if dist.get_rank() == 0:
        config_path = os.path.join(cfg.OUT_ROOT_DIR, "config.json")
        with open(config_path, "w") as f:
            f.write(cfg.dump())
        logger.info(f"Export training configuration as json: {config_path}")

    datasets_dict = {"Matterport3D": M3DDatasetAug,
                     "Stanford3D": StanfordDatasetAug,
                     "PanoSUNCG3D": PanoSUNCGDatasetAug,
                     "3D60": ThreeD60DatasetAug,
                     "Pano3D": Pano3DAugDataset}

    dataset = datasets_dict[cfg.DATA.DATASET_NAME]
    dataset_train = dataset(args.dataset_root_dir, args.train_data_list, cfg,
                            do_augmentation=True, mode="train")
    dataset_val = dataset(args.dataset_root_dir, args.valid_data_list, cfg,
                          do_augmentation=False,  mode="eval")

    writers_train = SummaryWriter(out_logging_train)
    writers_val = SummaryWriter(out_logging_val)

    pretrained=True
    encoder_model_dict = {"swin": SwinSphDecoderNet,
                          "resNet": ResnetSphDecoderNet,
                          "effnet": EffnetSphDecoderNet}
    model_type = encoder_model_dict[cfg.BACKBONE.TYPE]
    model = model_type(cfg, pretrained=pretrained)
    model.to(device)
    if cfg.TRAIN.PRETRAINED_MODEL != '':
        logger.info(f"Use pre-downloaded pretrained model: {cfg.TRAIN.PRETRAINED_MODEL}")

    data_loader_train = init_loader(cfg, dataset_train, is_train=True)
    data_loader_val = init_loader(cfg, dataset_val, is_train=False)
    optimizer = init_optimizer(cfg, model)
    if cfg.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False,
                                                      find_unused_parameters=True)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")


    model_without_ddp = model.module

    lr_scheduler = init_scheduler(cfg, optimizer, len(data_loader_train))
    logger.info(f"Use learning rate scheduler: {cfg.TRAIN.LR_SCHEDULER.NAME}")

    best_rel_err = float('inf')
    best_rel_err_epoch = -1

    # resume trainning process from checkpoint
    if cfg.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(out_models)
        if resume_file is not None:
            if cfg.TRAIN.RESUME != '':
                logger.info(f"Auto-resume changing resume file from {cfg.TRAIN.RESUME} to {resume_file}")
            cfg.defrost()
            cfg.TRAIN.RESUME = resume_file
            cfg.freeze()
            logger.info(f'Auto resuming from {resume_file}')
        else:
            logger.info(f'No checkpoint found in {out_models}, ignoring auto resume')

    if cfg.TRAIN.RESUME:
        rel_err = load_checkpoint_file(cfg, model_without_ddp, optimizer, lr_scheduler)
        if best_rel_err > rel_err:
            best_rel_err = rel_err
            best_rel_err_epoch = cfg.TRAIN.START_EPOCH - 1
        logger.info("Validation for auto-resumed model")
        validate(cfg, data_loader_val, model, device)
        logger.info("Auto resume and validation done!")
    logger.info("Start training proccess")


    start_time = time.time()
    step = 0
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(cfg, model, data_loader_train, optimizer, epoch, lr_scheduler, device, writers_train)
        eval_metrics = validate(cfg, data_loader_val, model, device)

        # writing validation events
        for l, v in eval_metrics.items():
            writers_val.add_scalar("{}".format(l), v, epoch)

        # choose the best model with minimum relative error metric
        rel_err = eval_metrics["err/abs_rel"]
        if best_rel_err > rel_err:
            best_rel_err = rel_err
            best_rel_err_epoch = epoch
        logger.info(f'Best relative error: {best_rel_err:.4f}, best relative error epoch: {best_rel_err_epoch}')
        if dist.get_rank() == 0 and (epoch % cfg.SAVE_FREQ == 0 or epoch == (cfg.TRAIN.EPOCHS - 1)):
            save_checkpoint(cfg, epoch, model_without_ddp, rel_err, optimizer, lr_scheduler, out_models)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Total training time {}'.format(total_time_str))

