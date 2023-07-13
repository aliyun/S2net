from __future__ import absolute_import, division, print_function
import os, argparse, tqdm
import matplotlib.pyplot as plt

import cv2
import healpy as hp
import numpy as np
import open3d as o3d


def read_list(list_file):
    file_pair_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            file_pair_list.append(line.strip().split(" "))
    return file_pair_list


def save_pcd(xyz, rgb, out_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)


class SphericalPixelization:
    def __init__(self, nside):
        assert hp.isnsideok(nside), "'nside' should be generally a power of 2"
        self.nside = nside
        self.pixel_num_sp = hp.nside2npix(self.nside)
        self.pixel_idx = np.arange(self.pixel_num_sp)
        self.sp_xyz = hp.pix2vec(self.nside, self.pixel_idx, nest=True)
        self.sp_ll = hp.pix2ang(self.nside, self.pixel_idx, nest = True, lonlat = True)

    def get_sphere_pixel_color(self, img, use_bilinear_interp = False):
        assert len(img.shape) == 3, "Only support equi-rectangular color image mapping"
        h, w, c = img.shape
        x, y = self.sp_ll[0] / 360.0 * w, (self.sp_ll[1] + 90.0) / 180.0 * h
        # retrieve colors from equi-rectangular image
        if use_bilinear_interp:
            x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
            x1, y1 = x0 + 1, y0 + 1
            x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
            x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)
            wa = np.expand_dims(wa, 1).repeat(3, axis = 1)
            wb = np.expand_dims(wb, 1).repeat(3, axis = 1)
            wc = np.expand_dims(wc, 1).repeat(3, axis = 1)
            wd = np.expand_dims(wd, 1).repeat(3, axis = 1)
            rgb = (wa * img[y0, x0, :] + wb * img[y1, x0, :] +
                   wc * img[y0, x1, :] + wd * img[y1, x1, :]).astype(np.uint8)
        else:
            x, y = np.around(x).astype(np.int32), np.around(y).astype(np.int32)
            x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
            rgb = img[y, x, :]
        return rgb

    def get_sphere_pixel_xyz(self):
        return np.stack([self.sp_xyz[0], self.sp_xyz[1], self.sp_xyz[2]], axis=1)

    def get_sphere_pixel_depth(self, depth_img):
        assert len(depth_img.shape) == 2, "Only support equi-rectangular depth image mapping"
        h, w = depth_img.shape
        x, y = self.sp_ll[0] / 360.0 * w, (self.sp_ll[1] + 90.0) / 180.0 * h
        x, y = np.around(x).astype(np.int32), np.around(y).astype(np.int32)
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        return depth_img[y, x]


def main():
    # argument parser
    parser = argparse.ArgumentParser(
        description = 'convert (equi-rectangular)panorama color and depth images to heapix map')

    parser.add_argument('dataset_root', type=str, default='', help='dataset root directory')
    parser.add_argument('rgb_depth_list', type=str, default='', help='relative path list for color and depth images')
    parser.add_argument('output_dir', type=str, default='', help='the output directory')
    parser.add_argument('--nside', type=int, default=128, help='heapix nside, pixels = 12 x nside^2')
    parser.add_argument('--depth_scale', type=float, default=4000.0, help='depth scale')
    parser.add_argument('--save_vis_sphere', action='store_true', help='save sphere pdc for visualization')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_rgb_npy_dir = os.path.join(args.output_dir, "rgb_npy")
    out_depth_npy_dir = os.path.join(args.output_dir, "depth_npy")
    os.makedirs(out_rgb_npy_dir, exist_ok=True)
    os.makedirs(out_depth_npy_dir, exist_ok=True)
    if args.save_vis_sphere:
        out_sphere_vis_dir = os.path.join(args.output_dir, "vis_pcd")
        os.makedirs(out_sphere_vis_dir, exist_ok=True)

    cmap = plt.get_cmap('magma_r')

    rgb_depth_path_list = read_list(args.rgb_depth_list)
    sp = SphericalPixelization(args.nside)
    print('=> Converting color panorama image into healpix map...')
    pbar = tqdm.tqdm(rgb_depth_path_list)
    for image_path in pbar:
        rgb_path, depth_path = image_path[0], image_path[1]

        rgb_path = os.path.join(args.dataset_root, rgb_path)
        depth_path = os.path.join(args.dataset_root, depth_path)

        rgb_base_name = os.path.basename(rgb_path).split(".")[0]
        depth_base_name = os.path.basename(depth_path).split(".")[0]

        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        d_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        sp_rgb = sp.get_sphere_pixel_color(rgb_img)
        sp_depth = sp.get_sphere_pixel_depth(d_img)
        out_sp_rgb_path = os.path.join(out_rgb_npy_dir, rgb_base_name + ".npy")
        out_sp_depth_path = os.path.join(out_depth_npy_dir, depth_base_name + ".npy")
        np.save(out_sp_rgb_path, sp_rgb)
        np.save(out_sp_depth_path, sp_depth / args.depth_scale)
        if args.save_vis_sphere:
            sp_xyz = sp.get_sphere_pixel_xyz()
            out_pcd_rgb_path = os.path.join(out_sphere_vis_dir, rgb_base_name + "_color_sp.ply")
            save_pcd(sp_xyz, sp_rgb / 255.0, out_pcd_rgb_path)
            max_depth = 10.0
            rgb_map_from_depth = cmap(sp_depth / args.depth_scale / max_depth)
            out_pcd_depth_path = os.path.join(out_sphere_vis_dir, depth_base_name + "_depth_sp.ply")
            save_pcd(sp_xyz, rgb_map_from_depth[:, :3], out_pcd_depth_path)
            sp_depth_test = sp_depth.reshape((-1, 1)).repeat(3, axis=1) / args.depth_scale
            out_pcd_depth_path_test = os.path.join(out_sphere_vis_dir, depth_base_name + "_3D.ply")
            save_pcd(sp_xyz * sp_depth_test, sp_rgb / 255.0, out_pcd_depth_path_test)

    print("=> Convert dataset to heapix map done!")


if __name__ == "__main__":
    main()