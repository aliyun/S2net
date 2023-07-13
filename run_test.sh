export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 python3 run_sp_monodepth_infer_eval.py --eval_depth_map \
-d /mnt/cap/limeng/sp_transformer/Matterport3D \
-i /mnt/cap/limeng/sp_transformer/Matterport3D/mpbench/matterport3d_test.txt \
-o /mnt/cap/WSB/Code/depth_final_test/out/test_110/ \
-c /mnt/cap/WSB/Code/depth_final_test/depth_estimation/config/test.yaml \
-m /mnt/cap/WSB/Code/depth_final_test/out/models/ckpt_epoch_110.pth