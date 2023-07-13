#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import open3d as o3d
import math, cv2


def interpolate_bilinear(tgt_theta, tgt_phi, src_rgb):
    src_height = src_rgb.shape[0]
    src_width = src_rgb.shape[1]
    coor_r = ((tgt_theta + 0.5 * np.pi) / np.pi * src_height).astype(np.uint32)
    coor_c = (0.5 * (tgt_phi + np.pi) / np.pi * src_width).astype(np.uint32)
    return coor_r, coor_c

def generate_sphere_pointcloud(rgb, res_lat, res_lon):
    lat_samples = np.int32(np.pi / res_lat)
    lon_samples = np.int32(2 * np.pi / res_lon)
    # lattitude angle
    theta = np.arange(lat_samples).reshape(lat_samples, 1) * res_lat + res_lat * 0.5 - np.pi / 2
    theta = np.repeat(theta, lon_samples, axis=1)
    # longitude
    phi = np.arange(lon_samples).reshape(1, lon_samples) * res_lon + res_lon * 0.5 - np.pi
    phi = np.repeat(phi, lat_samples, axis=0)

    X = (np.cos(theta) * np.cos(phi)).flatten()
    Y = (np.sin(theta)).flatten()
    Z = (np.cos(theta) * np.sin(phi)).flatten()
    XYZ = np.stack([X, Y, Z], axis=1)
    map1, map2 = interpolate_bilinear(theta, phi, rgb)
    R = rgb[map1, map2, 0].flatten() / 255.0
    G = rgb[map1, map2, 1].flatten() / 255.0
    B = rgb[map1, map2, 2].flatten() / 255.0

    RGB = np.stack([R, G, B], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ)
    pcd.colors = o3d.utility.Vector3dVector(RGB)
    return pcd


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description = 'this script batch-produces pointclouds from depth images list')
    parser.add_argument('pano_image_path', type = str, default = '', help = 'Input panorama image path')
    parser.add_argument('output_sphere', type = str, default = '', help = 'Output point cloud sphere with unit radius')
    parser.add_argument('--angle_resolution', type = float, default='0.09', help='angle resolution(degree)')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output_sphere)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rgb = cv2.imread(args.pano_image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb is None:
        print("Load panorama image failed: {}".format(args.pano_image_path))
        exit(-1)
    else:
        pcd = generate_sphere_pointcloud(rgb, args.angle_resolution / 180.0 * np.pi, args.angle_resolution / 180.0 * np.pi)
        o3d.io.write_point_cloud(args.output_sphere, pcd)

