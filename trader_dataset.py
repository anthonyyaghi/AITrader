import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage import io, transform
import pandas as pd


class TraderDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        # image = Image.open(img_path)
        image = io.imread(img_path)
        actions = int(self.df.iloc[idx, 1:])

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, actions


def compute_mean_std(dataset):
    data_r = np.dstack([dataset[i][0][:, :, 0] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][0][:, :, 1] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][0][:, :, 2] for i in range(len(dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std
