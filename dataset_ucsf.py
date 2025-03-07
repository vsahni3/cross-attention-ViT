
import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset

import os
import numpy as np

import pandas as pd
from scipy import ndimage



from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd
)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityd, 
    ResizeWithPadOrCropd, RandFlipd, RandRotate90d, RandAffined, RandCoarseShuffled,
    RandGaussianNoised, RandAdjustContrastd, RandGaussianSmoothd, RandZoomd,
    RandCoarseDropoutd, ToTensord
)
from monai.transforms.compose import MapTransform
from monai.config import print_config
from monai.metrics import DiceMetric
from sklearn.model_selection import train_test_split
import pandas as pd 

import copy






# # Only be careful about the ResizeWithPadOrCropd. I am not sure should you use it or not. In my case,
# # I need a volume with fixed size.

# # One more thing, be careful of the normalization, CT is quantative and MRI is not, so they need different normalization here.
# # Maybe not your case.

class BrainDataset(Dataset):
    def __init__(self, data, config, types=("T1c", "T2"), is_train=True, folder='ucsf-data'):
        self.target = config.target
        self.data = data
        self.types = types
        self.is_train = is_train
        self.folder = folder

        self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image"],reader='nibabelreader'),
                    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    # Orientationd(keys=["image"], axcodes="RAS"),
                    # Spacingd(
                    #     keys=["image"],
                    #     pixdim=config.spacing,
                    #     mode=("bilinear"),
                    # ),
                    # CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
                    # ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
                    ResizeWithPadOrCropd(
                          keys=["image"],
                          spatial_size=config.img_size,
                          constant_values = -1,
                    ),
                    
                    #augmentations to prevent overfitting
                    # RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # Horizontal Flip
                    # RandRotate90d(keys=["image"], prob=0.2, max_k=1),  # Slight 90-degree rotation
                    # RandAffined(keys=["image"], prob=0.2, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),  

                    # # Contrast and noise augmentations (Improve generalization)
                    # RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),  # Contrast jittering
                    # RandGaussianNoised(keys=["image"], prob=0.2, mean=0, std=0.1),  # Add Gaussian noise
                    # RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5)),  # Blur for domain adaptation

                    # # CutMix-style augmentation (Random shuffle blocks, approximating CutMix)
                    # RandCoarseShuffled(keys=["image"], prob=0.2, holes=5, spatial_size=(20, 20, 20)),  

                    # # Mixup Alternative: Coarse dropout (Mimics Mixup-like occlusions)
                    # RandCoarseDropoutd(keys=["image"], prob=0.2, holes=3, spatial_size=(15, 15, 15), fill_value=-1),

                    # # Random zoom for scale invariance
                    # RandZoomd(keys=["image"], prob=0.2, min_zoom=0.9, max_zoom=1.1),
                    
                    ToTensord(keys=["image"]),
                ]
            )
        self.test_transforms = Compose(
                [
                    LoadImaged(keys=["image"],reader='nibabelreader'),
                    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    # Orientationd(keys=["image"], axcodes="RAS"),
                    # Spacingd(
                    #     keys=["image"],
                    #     pixdim=config.spacing,
                    #     mode=("bilinear"),
                    # ),
                    # CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
                    # ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
                    ResizeWithPadOrCropd(
                          keys=["image"],
                          spatial_size=config.img_size,
                          constant_values = -1,
                    ),
                    ToTensord(keys=["image"]),
                ]
            ) 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        case_id = row['ID']
        target = int(row[self.target])
        data = []
        for mri_type in self.types:

            img_path = f"{self.folder}/{case_id}_nifti/{case_id}_{mri_type}.nii.gz"
            cao = {"image": img_path}
            transform = self.train_transforms if self.is_train else self.test_transforms
            affined_data_dict = transform(cao)   
            img_tensor = affined_data_dict['image'].to(torch.float)
            data.append(img_tensor)
        return torch.stack(data), torch.tensor(target, dtype=torch.long)

def clean_data(data, target):
    to_drop = ['138', '181', '175', '278', '289', '315']
    pattern = '|'.join(to_drop)
    data = data[~data['ID'].str.contains(pattern)]
    data.loc[:, 'ID'] = data['ID'].apply(lambda x: '-'.join([*x.split('-')[:-1], x.split('-')[-1].zfill(4)]))
    
    data = data[~((data[target] == 'indeterminate') | (data[target].isna()))]
    data[target] = (data[target] == 'positive').astype(float)
    return data



