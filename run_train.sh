export OMP_NUM_THREADS=8
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank 0 run_sp_monodepth_train.py \
-d /path/to/Matterport3D \
-i /path/to/Matterport3D/matterport3d_train_file_list.txt \
-v /path/to/Matterport3D/eval_file_list.txt \
-o /path/to/output_folder \
-c /path/to/config/train_cfg_2cards.yaml \
-p /path/to/swin_backbone/swin_base_patch4_window7_224_22k.pth
