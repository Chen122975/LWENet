
import os
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset
from PIL import Image
class DatasetPair(Dataset):
    def __init__(self, cover_dir, stego_dir,
                 transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.cover_list = os.listdir(cover_dir)
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0,1], dtype='int32')
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])
        cover = Image.open(cover_path)
        images = np.empty((2, cover.size[0], cover.size[1], 1),
                          dtype='uint8')
        images[0,:,:,0] = np.array(cover)
        stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])
        stego = Image.open(stego_path)
        images[1,:,:,0] = np.array(stego)
        samples = {'images': images, 'labels': labels}
        if self.transform:
            samples = self.transform(samples)
        return samples
class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        images = (images.transpose((0,3,1,2)).astype('float32') / 255)
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels).long()}


class AugData():
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']

        # Rotation
        rot = random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        # Mirroring
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()

        new_sample = {'images': images, 'labels': labels}

        return new_sample
