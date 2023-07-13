#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import open3d as o3d


def convert_to_pointclouds(o3d_img, fx, fy, cx, cy, scale, inv_depth, max_d, min_d):
    np_image = np.array(o3d_img)
    height = np_image.shape[0]
    width = np_image.shape[1]
    if inv_depth:
        pcd = o3d.geometry.PointCloud()
        inv_min = 1.0 / max_d * scale
        inv_max = 1.0 / min_d * scale
        points = []
        for row in range(height):
            for col in range(width):
                inv_d = np_image[row][col]
                if inv_d > inv_max or inv_d < inv_min:
                    continue
                xyz = [scale / inv_d * (col - cx) / fx, scale / inv_d * (row - cy) / fy, scale / inv_d]
                points.append(xyz)
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        return pcd
    else:
        cam_mat = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth = o3d_img, intrinsic = cam_mat,
                                                                    depth_scale = scale, depth_trunc = max_d)
        return o3d_cloud


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description = 'this script batch-produces pointclouds from depth images list')
    parser.add_argument('depth_image_list', type = str, default = '', help = '')
    parser.add_argument('output_ws', type = str, default = '', help = '')
    parser.add_argument('--fx', type = float, default = '-1.0', help = 'specify focal length for normal depth image')
    parser.add_argument('--fy', type = float, default = '-1.0', help = 'specify focal length for normal depth image')
    parser.add_argument('--cx', type = float, default = '-1.0', help = 'specify principal points for normal depth image')
    parser.add_argument('--cy', type = float, default = '-1.0', help = 'specify principal points for normal depth image')
    parser.add_argument('--scale', type = float, default='1000', help='specify depth scale')
    parser.add_argument('--max_depth', type = float, default='10.0', help='maximum depth threshold')
    parser.add_argument('--min_depth', type = float, default='0.5', help='minimum depth threshold')
    parser.add_argument('--inv_depth', dest='inv_depth', action='store_true', help='inverse depth')
    parser.set_defaults(inv_depth=False)
    args = parser.parse_args()
    depth_image_list = args.depth_image_list
    output_ws = args.output_ws
    if not os.path.exists(output_ws):
        os.makedirs(output_ws)
    assert args.fx > 0.0 and args.fy > 0.0 and args.cx > 0.0 and args.cy > 0.0
    with open(depth_image_list) as image_lists:
        for image_path in image_lists:
            image_path = image_path.strip('\n')
            base_name = os.path.basename(image_path).split(".")[0]
            out_image_path = os.path.join(output_ws, base_name + '.ply')
            img_o3d = o3d.io.read_image(image_path)
            if img_o3d is None:
                print("Load image failed: {}".format(image_path))
                continue
            else:
                pcd = convert_to_pointclouds(img_o3d, args.fx, args.fy, args.cx, args.cy,
                                             args.scale, args.inv_depth, args.max_depth, args.min_depth)
                print(out_image_path)
                o3d.io.write_point_cloud(out_image_path, pcd)