import os, random
import numpy as np
import cv2
import healpy as hp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
from torch.utils import data
from torchvision import transforms
import torch.distributed as dist

from .decoder import SphPixelization
from .util_helper import to_numpy


def read_list_from_file(list_file, is_pair=True):
    target_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line != "":
                target_list.append(line.split(" ") if is_pair else line)
    return target_list


def compose_transform(cfg, do_augmentation=True, is_training=True):
    t = []
    if is_training and do_augmentation:
        color_jitter = transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS,
                                              contrast=cfg.AUG.CONTRAST,
                                              saturation=cfg.AUG.SATURATION,
                                              hue=cfg.AUG.HUE)
        t.append(transforms.ToPILImage())
        t.append(transforms.RandomApply(torch.nn.ModuleList([color_jitter]), p=0.5))
    t.append(transforms.ToTensor())  # range [0, 255] -> [0.0,1.0]
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


# steal from swin
class SubsetRandomSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def init_loader(cfg, dataset, is_train=True):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if is_train:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=True,
        )
    else:
        indices = np.arange(dist.get_rank(), len(dataset), dist.get_world_size())
        sampler = SubsetRandomSampler(indices)
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=cfg.TESTING.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=False
            )
    return data_loader


def compute_spp_index_map(nside=128, img_size=(512, 1024)):
    h, w = img_size
    pixel_num_sp = hp.nside2npix(nside)
    pixel_idx = np.arange(pixel_num_sp)
    sp_ll = hp.pix2ang(nside, pixel_idx, nest=True, lonlat=True)
    x, y = sp_ll[0] / 360.0 * w, (sp_ll[1] + 90.0) / 180.0 * h
    x, y = np.around(x).astype(np.int32), np.around(y).astype(np.int32)
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    return torch.Tensor(x + y * w).long()


class M3DDatasetAug(data.Dataset):
    def __init__(self, root_dir, list_file, cfg, do_augmentation=True, mode="train",
                 keep_raw_rgb=False, keep_raw_gt_depth=False):
        super(M3DDatasetAug, self).__init__()
        self.root_dir = root_dir
        self.is_training_or_val = (mode == "train" or mode == "eval")
        self.file_list = read_list_from_file(list_file, self.is_training_or_val)
        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.augmentation = do_augmentation
        self.keep_raw_rgb = keep_raw_rgb
        self.keep_raw_gt_depth = keep_raw_gt_depth
        self.to_tensor_normalize = compose_transform(cfg, do_augmentation=False)
        self.color_jitter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS,
                                   contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION,
                                   hue=cfg.AUG.HUE)])
        self.spp_index_map = compute_spp_index_map(nside=128, img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        rgb_path = os.path.join(self.root_dir,
                                self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)

        # if dataset for inferring
        if not self.is_training_or_val:
            mask_border_range = 80
            rgb[0:mask_border_range, :, :] = 0
            rgb[-mask_border_range:, :, :] = 0
        if self.keep_raw_rgb:
            inputs["raw_rgb"] = rgb.copy()
        if self.is_training_or_val:
            depth_path = os.path.join(self.root_dir, self.file_list[idx][1])
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            # To keep gt depth is consistent with rgb (no misalignment)
            gt_depth = cv2.resize(gt_depth, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST)

            gt_depth = gt_depth.astype(np.float) / 4000.0
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth.copy(), axis=0))
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            # random yaw rotation
            roll_idx = random.randint(0, self.cfg.DATA.IMG_WIDTH)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_jitter(rgb))
        inputs["rgb"] = self.to_tensor_normalize(rgb)
        if self.is_training_or_val:
            gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
            inputs["gt_depth"] = gt_depth.reshape(1, -1).index_select(dim=-1, index=self.spp_index_map).unsqueeze(0)
            inputs["mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["gt_depth"])))
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth_mask"] = ((inputs["raw_gt_depth"] > 0) & (inputs["raw_gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["raw_gt_depth"])))
        inputs["name"] = os.path.splitext(
            os.path.basename(self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx]))[0]

        return inputs


class StanfordDatasetAug(data.Dataset):
    def __init__(self, root_dir, list_file, cfg, do_augmentation=True,
                 mode="train", keep_raw_rgb=False, keep_raw_gt_depth=False):
        super(StanfordDatasetAug, self).__init__()
        self.root_dir = root_dir
        self.is_training_or_val = (mode == "train" or mode == "eval")
        self.file_list = read_list_from_file(list_file, self.is_training_or_val)
        if mode == "train":
            self.file_list = 10 * self.file_list

        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.augmentation = do_augmentation
        self.keep_raw_rgb = keep_raw_rgb
        self.keep_raw_gt_depth = keep_raw_gt_depth
        self.to_tensor_normalize = compose_transform(cfg, do_augmentation=False)
        self.color_jitter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS,
                                   contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION,
                                   hue=cfg.AUG.HUE)])
        self.spp_index_map = compute_spp_index_map(nside=128, img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        rgb_path = os.path.join(self.root_dir,
                                self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)

        if self.keep_raw_rgb:
            inputs["raw_rgb"] = rgb.copy()
        if self.is_training_or_val:
            depth_path = os.path.join(self.root_dir, self.file_list[idx][1])
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            # To keep gt depth is consistent with rgb (no misalignment)
            gt_depth = cv2.resize(gt_depth, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST)

            gt_depth = gt_depth.astype(np.float) / 512
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth.copy(), axis=0))
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            # random yaw rotation
            roll_idx = random.randint(0, self.cfg.DATA.IMG_WIDTH)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_jitter(rgb))
        inputs["rgb"] = self.to_tensor_normalize(rgb)
        if self.is_training_or_val:
            gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
            inputs["gt_depth"] = gt_depth.reshape(1, -1).index_select(dim=-1, index=self.spp_index_map).unsqueeze(0)
            inputs["mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["gt_depth"])))
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth_mask"] = ((inputs["raw_gt_depth"] > 0) & (inputs["raw_gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["raw_gt_depth"])))
        inputs["name"] = os.path.splitext(
            os.path.basename(self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx]))[0]

        return inputs


class PanoSUNCGDatasetAug(data.Dataset):
    def __init__(self, root_dir, list_file, cfg, do_augmentation=True,
                 mode="train", keep_raw_rgb=False, keep_raw_gt_depth=False):
        super(PanoSUNCGDatasetAug, self).__init__()
        self.root_dir = root_dir
        self.is_training_or_val = (mode == "train" or mode == "eval")
        self.file_list = read_list_from_file(list_file, self.is_training_or_val)
        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.augmentation = do_augmentation
        self.keep_raw_rgb = keep_raw_rgb
        self.keep_raw_gt_depth = keep_raw_gt_depth
        self.to_tensor_normalize = compose_transform(cfg, do_augmentation=False)
        self.color_jitter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS,
                                   contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION,
                                   hue=cfg.AUG.HUE)])
        self.spp_index_map = compute_spp_index_map(nside=128, img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        rgb_path = os.path.join(self.root_dir,
                                self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)

        if self.keep_raw_rgb:
            inputs["raw_rgb"] = rgb.copy()
        if self.is_training_or_val:
            depth_path = os.path.join(self.root_dir, self.file_list[idx][1])
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            # To keep gt depth is consistent with rgb (no misalignment)
            gt_depth = cv2.resize(gt_depth, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST)

            gt_depth = gt_depth.astype(np.float) * 10.0 / 255
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth.copy(), axis=0))
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            # random yaw rotation
            roll_idx = random.randint(0, self.cfg.DATA.IMG_WIDTH)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_jitter(rgb))
        inputs["rgb"] = self.to_tensor_normalize(rgb)
        if self.is_training_or_val:
            gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
            inputs["gt_depth"] = gt_depth.reshape(1, -1).index_select(dim=-1, index=self.spp_index_map).unsqueeze(0)
            inputs["mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["gt_depth"])))
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth_mask"] = ((inputs["raw_gt_depth"] > 0) & (inputs["raw_gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["raw_gt_depth"])))
        inputs["name"] = os.path.splitext(
            os.path.basename(self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx]))[0]

        return inputs


def recover_filename(file_name):

    splits = file_name.split('.')
    rot_ang = splits[0].split('_')[-1]
    file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]

    return file_name, int(rot_ang)


class ThreeD60DatasetAug(data.Dataset):
    def __init__(self, root_dir, list_file, cfg, do_augmentation=True,
                 mode="train", keep_raw_rgb=False, keep_raw_gt_depth=False):
        super(ThreeD60DatasetAug, self).__init__()
        self.root_dir = root_dir
        self.is_training_or_val = (mode == "train" or mode == "eval")
        self.file_list = read_list_from_file(list_file, self.is_training_or_val)
        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.augmentation = do_augmentation
        self.keep_raw_rgb = keep_raw_rgb
        self.keep_raw_gt_depth = keep_raw_gt_depth
        self.to_tensor_normalize = compose_transform(cfg, do_augmentation=False)
        self.color_jitter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS,
                                   contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION,
                                   hue=cfg.AUG.HUE)])
        self.spp_index_map = compute_spp_index_map(nside=128, img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
        self.cfg = cfg

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        if self.is_training_or_val:
            rgb_path, rot_ang = recover_filename(os.path.join(self.root_dir, self.file_list[idx][0]))
        else:
            rgb_path = os.path.join(self.root_dir, self.file_list[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)

        if self.keep_raw_rgb:
            inputs["raw_rgb"] = rgb.copy()
        if self.is_training_or_val:
            depth_path, _ = recover_filename(os.path.join(self.root_dir, self.file_list[idx][1]))
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            # To keep gt depth is consistent with rgb (no misalignment)
            gt_depth = cv2.resize(gt_depth, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST)

            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth.copy(), axis=0))
        if self.is_training_or_val and self.augmentation:
            if random.random() > 0.5:
                # random yaw rotation
                roll_idx = random.randint(0, self.cfg.DATA.IMG_WIDTH // 4) + \
                           (self.cfg.DATA.IMG_WIDTH * rot_ang) // 360
            else:
                roll_idx = (self.cfg.DATA.IMG_WIDTH * rot_ang) // 360

            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_jitter(rgb))

        inputs["rgb"] = self.to_tensor_normalize(rgb)
        if self.is_training_or_val:
            gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
            inputs["gt_depth"] = gt_depth.reshape(1, -1).index_select(dim=-1, index=self.spp_index_map).unsqueeze(0)
            inputs["mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["gt_depth"])))
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth_mask"] = ((inputs["raw_gt_depth"] > 0) & (inputs["raw_gt_depth"] <= self.max_depth)
                                & (~torch.isnan(inputs["raw_gt_depth"])))
        inputs["name"] = os.path.splitext(
            os.path.basename(self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx]))[0]

        return inputs


def weight_by_latitude(h, w):
    # -pi/2 - pi/2
    theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2.0 - np.pi / 2.0
    theta = np.repeat(theta, w, axis=1)
    weight = torch.cos(torch.from_numpy(np.expand_dims(theta, axis=0)))
    return weight


class Pano3DAugDataset(data.Dataset):
    def __init__(self, root_dir, yaml_file, cfg, do_augmentation=True,
                 mode="train", keep_raw_rgb=False, keep_raw_gt_depth=False):
        super(Pano3DAugDataset, self).__init__()
        self.root_dir = root_dir
        self.is_training_or_val = (mode == "train" or mode == "eval")
        self.file_list = read_list_from_yaml(yaml_file)
        self.max_depth = cfg.MODEL.MAX_DEPTH
        self.augmentation = do_augmentation
        self.keep_raw_rgb = keep_raw_rgb
        self.keep_raw_gt_depth = keep_raw_gt_depth
        self.to_tensor_normalize = compose_transform(cfg, do_augmentation=False)
        self.color_jitter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS,
                                   contrast=cfg.AUG.CONTRAST,
                                   saturation=cfg.AUG.SATURATION,
                                   hue=cfg.AUG.HUE)])
        self.spp_index_map = compute_spp_index_map(nside=128, img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        rgb_path = os.path.join(self.root_dir,
                                self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)

        if self.keep_raw_rgb:
            inputs["raw_rgb"] = rgb.copy()
        if self.is_training_or_val:
            depth_path = os.path.join(self.root_dir, self.file_list[idx][1])
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            # To keep gt depth is consistent with rgb (no misalignment)
            gt_depth = cv2.resize(gt_depth, dsize=(self.cfg.DATA.IMG_WIDTH, self.cfg.DATA.IMG_HEIGHT),
                                  interpolation=cv2.INTER_NEAREST)

            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth.copy(), axis=0))
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            # random yaw rotation
            roll_idx = random.randint(0, self.cfg.DATA.IMG_WIDTH)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
        if self.is_training_or_val and self.augmentation and random.random() > 0.5:
            rgb = np.asarray(self.color_jitter(rgb))
        inputs["rgb"] = self.to_tensor_normalize(rgb)
        if self.is_training_or_val:
            gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
            inputs["gt_depth"] = gt_depth.reshape(1, -1).index_select(dim=-1, index=self.spp_index_map).unsqueeze(0)
            inputs["mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth)
                              & (~torch.isnan(inputs["gt_depth"])))
            if self.keep_raw_gt_depth:
                inputs["raw_gt_depth_mask"] = ((inputs["raw_gt_depth"] > 0) & (inputs["raw_gt_depth"] <= self.max_depth)
                                               & (~torch.isnan(inputs["raw_gt_depth"])))
        inputs["name"] = os.path.splitext(
            os.path.basename(self.file_list[idx][0] if self.is_training_or_val else self.file_list[idx]))[0]
        if self.mode == "eval":
            inputs["eval_weight"] = weight_by_latitude(self.cfg.DATA.IMG_HEIGHT, self.cfg.DATA.IMG_WIDTH)
        return inputs
