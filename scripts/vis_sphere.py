from __future__ import absolute_import, division, print_function
import os, argparse
import healpy as hp
import numpy as np
from convert_healpix import save_pcd


def main():
    # argument parser
    parser = argparse.ArgumentParser(description='visualize color sphere')
    parser.add_argument('input_rgb_npy', type=str, default='', help='input rgb pytorch npy file')
    parser.add_argument('output_ply', type=str, default='', help='output ply file path')
    parser.add_argument('--nside', type=int, default=128, help='heapix nside, pixels = 12 x nside^2')

    args = parser.parse_args()
    pixel_num_sp = hp.nside2npix(args.nside)
    rgb_hp = np.load(args.input_rgb_npy)

    rgb_hp = np.transpose(rgb_hp, (2, 1, 0)).squeeze(1)
    w, c = rgb_hp.shape
    assert w == pixel_num_sp and c == 3,  "Inconsistent dims!"
    pixel_idx = np.arange(pixel_num_sp)
    sp_xyz = hp.pix2vec(args.nside, pixel_idx, nest=True)
    sp_xyz = np.stack([sp_xyz[0], sp_xyz[1], sp_xyz[2]], axis=1)
    output_path = args.output_ply
    save_pcd(sp_xyz, rgb_hp/255.0, output_path)


if __name__ == "__main__":
    main()