#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import open3d as o3d
import math, cv2


def convert_to_pointclouds_pano(depth_img, scale, inv_depth, max_d, min_d):
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    assert width / height == 2.0
    pcd = o3d.geometry.PointCloud()
    points = []
    for row in range(height):
        for col in range(width):
            depth = depth_img[row][col] / scale
            if inv_depth:
                depth = 1.0 / depth
            if depth > max_d or depth < min_d:
                continue
            theta = math.pi * ((row + 0.5) / height - 0.5)
            fai = 2.0 * math.pi * (col + 0.5) / width - math.pi
            xyz = [math.cos(theta) * math.cos(fai) * depth, math.sin(theta) * depth, math.cos(theta) * math.sin(fai) * depth]
            points.append(xyz)
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description = 'this script batch-produces pointclouds from depth images list')
    parser.add_argument('depth_image_list', type = str, default = '', help = '')
    parser.add_argument('output_ws', type = str, default = '', help = '')
    parser.add_argument('--scale', type = float, default=4000.0, help='specify depth scale')
    parser.add_argument('--min_depth', type = float, default='0.5', help='maximum depth threshold')
    parser.add_argument('--max_depth', type = float, default='10.0', help='minimum depth threshold')
    parser.add_argument('--inv_depth', dest='inv_depth', action='store_true', help='inverse depth')
    parser.set_defaults(inv_depth=False)
    args = parser.parse_args()
    depth_image_list = args.depth_image_list
    output_ws = args.output_ws
    if not os.path.exists(output_ws):
        os.makedirs(output_ws)
    with open(depth_image_list) as image_lists:
        for image_path in image_lists:
            image_path = image_path.strip('\n')
            base_name = os.path.basename(image_path).split(".")[0]
            out_image_path = os.path.join(output_ws, base_name + ".ply")
            cv_img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            if cv_img is None:
                print("Load image failed: {}".format(image_path))
                continue
            else:
                pcd = convert_to_pointclouds_pano(cv_img, args.scale, args.inv_depth,
                                                  args.max_depth, args.min_depth)
                o3d.io.write_point_cloud(out_image_path, pcd)
