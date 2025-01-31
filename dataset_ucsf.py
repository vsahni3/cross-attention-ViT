
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
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensity, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, EnsureTyped)
from monai.transforms.compose import MapTransform
from monai.config import print_config
from monai.metrics import DiceMetric
from sklearn.model_selection import train_test_split
import pandas as pd 

import copy
import config

# Load configuration
config = config.get_mgmt_config()





# # Only be careful about the ResizeWithPadOrCropd. I am not sure should you use it or not. In my case,
# # I need a volume with fixed size.

# # One more thing, be careful of the normalization, CT is quantative and MRI is not, so they need different normalization here.
# # Maybe not your case.

class BrainDataset(Dataset):
    def __init__(self, data, target=config.target, types=("T1c", "T2"), is_train=True, folder='ucsf-data'):
        self.target = target
        self.data = data
        self.types = types
        self.is_train = is_train
        self.folder = folder

        self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image"],reader='nibabelreader'),
                    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(
                        keys=["image"],
                        pixdim=config.spacing,
                        mode=("bilinear"),
                    ),
                    # CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
                    ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
                    ResizeWithPadOrCropd(
                          keys=["image"],
                          spatial_size=config.img_size,
                          constant_values = -1,
                    ),
                    
                    #augmentations to prevent overfitting
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # Flip horizontally
                    RandRotate90d(keys=["image"], prob=0.3, max_k=1),  # Rotate slightly
                    RandAffined(keys=["image"], prob=0.3, rotate_range=(0.05, 0.05, 0.05), scale_range=(0.1, 0.1, 0.1)),  
                    ToTensord(keys=["image"]),
                ]
            )
        self.test_transforms = Compose(
                [
                    LoadImaged(keys=["image"],reader='nibabelreader'),
                    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    Spacingd(
                        keys=["image"],
                        pixdim=config.spacing,
                        mode=("bilinear"),
                    ),
                    # CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
                    ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
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




