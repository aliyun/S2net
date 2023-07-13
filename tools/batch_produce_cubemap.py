#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, argparse
import robust_exec
if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(description = 'this script batch-produces cubic map for all panorama image')
    parser.add_argument('pano_image_list', type = str, default = '', help = '')
    parser.add_argument('output_ws', type = str, default = '', help = '')
    parser.add_argument('--cube_size_pix', type = str, default = '1040', help = '')
    parser.add_argument('--cube_fov', type = str, default = '90', help = '')
    parser.add_argument('--src_width', type = str, default = '6720', help = '')
    parser.add_argument('--src_height', type = str, default = '3360', help = '')

    args = parser.parse_args()
    pano_image_list = args.pano_image_list
    output_ws = args.output_ws
    if not os.path.exists(output_ws):
        os.makedirs(output_ws)
    with open(pano_image_list) as image_lists:
        for image_name in image_lists:
            image_name = image_name.strip('\n')
            cmds = ['bash', 'nona_cube_faces.sh', image_name, args.cube_size_pix, args.cube_fov, args.src_width, args.src_height, output_ws]
            robust_exec.robust_exec(cmds)
