# datasets/sequence_folders.py
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
from .scared_dataset import SCAREDDataset

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """
    Soporte para datasets tipo KITTI, NYU y SCARED.
    Carga secuencias de longitud N (tgt + vecinos).
    """
    def __init__(self, root, seed=None, train=True, sequence_length=3,
                 transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.sequence_length = sequence_length

        if dataset.lower() == 'scared':
            # Usa los splits del folder (train.txt o val.txt)
            scene_list_path = self.root / ('train_files.txt' if train else 'val_files.txt')
            with open(scene_list_path, 'r') as f:
                filenames = f.read().splitlines()
            print(f"[SCARED] Loading {len(filenames)} filenames from {scene_list_path}")
            self.dataset_impl = SCAREDDataset(
                data_path=str(self.root),
                filenames=filenames,
                height=256,
                width=320,
                frame_idxs=[0, -1, 1],
                num_scales=1,
                is_train=train
            )
        else:
            # Resto de datasets (kitti, nyu)
            scene_list_path = self.root / ('train.txt' if train else 'val.txt')
            self.scenes = [self.root / folder[:-1] for folder in open(scene_list_path)]
            self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        demi_length = (self.sequence_length - 1) // 2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < self.sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs) - demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        if self.dataset.lower() == 'scared':
            return self.dataset_impl.__getitem__(index)
        else:
            sample = self.samples[index]
            tgt_img = load_as_float(sample['tgt'])
            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
            if self.transform is not None:
                imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                ref_imgs = imgs[1:]
            else:
                intrinsics = np.copy(sample['intrinsics'])
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        if self.dataset.lower() == 'scared':
            return len(self.dataset_impl)
        return len(self.samples)
