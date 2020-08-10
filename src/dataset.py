import pandas as pd
import torch
import numpy as np
from PIL import Image
import albumentations
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class FlowerDataset(Dataset):
    def __init__(self, image_path, targets, transform=None):
        self.image_path = image_path
        self.targets = targets
        self.transform = transform

        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)

        # self.augmentation = albumentations.Compose(
        #    [
        #        albumentations.Normalize(mean, std, always_apply=True),
        #        albumentations.HorizontalFlip(p=0.5)
        #    ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        targets = self.targets[index]

        if self.transform:
            for transform_item in self.transform:
                image = transform_item(image)

        return (image, targets)
