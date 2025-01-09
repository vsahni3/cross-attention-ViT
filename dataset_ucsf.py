# ways to change datalaoder:
# instead of fixed number of slices see if model can take in variable slices
# change how fewer than expected slices are handled, currently 0 padding 
# some cv2 stuff done during dicom loading 
# in load_dicom_image pixels are normalized to pixel_size x pixel_size, maybe model can accept varying sizes
# pixel values normalized between 0 and 1
# model experiment with flattening 2d slices and then passing in all tokens from all slices, vs getting 3d patches and flattening


import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Load the .nii.gz file
def load_nii_to_tensor(file_path):
    # Load the NIfTI file using nibabel
    nii_img = nib.load(file_path)
    print(nii_img)
    fe
    # Extract the image data as a NumPy array
    img_data = nii_img.get_fdata()  # Returns the image data as a 3D NumPy array

    # Convert the NumPy array to a PyTorch tensor
    tensor_data = torch.tensor(img_data, dtype=torch.float32)

    return tensor_data

# Example usage
file_path = '/Users/varunsahni/Downloads/test-data/UCSF-PDGM-0005_T1.nii.gz'  # Replace with your file path


import glob
import os
import re
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import cv2

from pydicom.uid import ExplicitVRLittleEndian
import argparse


import monai

import pandas as pd



import config
config = config.get_3DReg_config()
from sklearn.model_selection import train_test_split
# parser = argparse.ArgumentParser()
# parser.add_argument("--fold", default=0, type=int)
# parser.add_argument("--type", default="FLAIR", type=str)
# parser.add_argument("--model_name", default="b0", type=str)
# args = parser.parse_args()




device = torch.device("cuda")


class BrainDataset(Dataset):
    def __init__(
        self, data, target="MGMT status", types=("FLAIR", "T1", "T2"), is_train=True, do_load=True, folder='ucsf-data', use_indexes=True
    ):
        self.target = target
        self.data = data
        self._clean_data()
        self.types = types

        self.is_train = is_train
        self.folder = folder
        self.do_load = do_load
        self.use_indexes = use_indexes
        if self.use_indexes:
            self.img_indexes = self._prepare_biggest_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        case_id = row['ID']
        target = int(row[self.target])
        image = self.load_image(case_id)
        if self.is_train:
            return {"image": image, "target": target, "case_id": case_id}
        else:
            return {"image": image, "case_id": case_id}
    
    def _clean_data(self):
        to_drop = ['138', '181', '175', '278']
        pattern = '|'.join(to_drop)

        self.data = self.data[~self.data['ID'].str.contains(pattern)]
        self.data.loc[:, 'ID'] = self.data['ID'].apply(lambda x: '-'.join([*x.split('-')[:-1], x.split('-')[-1].zfill(4)]))
        self.data = self.data[~((self.data[self.target] == 'indeterminate') | (self.data[self.target].isna()))]
        self.data[self.target] = (self.data[self.target] == 'positive').astype(int)
        
    def _prepare_biggest_images(self):
        big_image_indexes = {}
        if (f"big_image_indexes.pkl" in os.listdir("."))\
            and (self.do_load):
            print("Loading the best images indexes for all the cases...")
            big_image_indexes = joblib.load(f"big_image_indexes.pkl")
            return big_image_indexes
        else:
            
            print("Calculating the best scans for every case...")
            bad = []
            for mri_type in self.types:
                for row in tqdm(self.data.iterrows(), total=len(self.data)):
                    case_id = row[1]['ID']
                    
                    
                    path = f"{self.folder}/{case_id}_nifti/{case_id}_{mri_type}.nii.gz"
                    file_tensor = load_nii_to_tensor(path)


                    brightness_vals = file_tensor.sum(dim=(0, 1))
                    if torch.all(file_tensor == 0):
                        middle = file_tensor.shape[-1] // 2
                    else:
                        middle = file_tensor.argmax()
                    big_image_indexes[(case_id, mri_type)] = middle

            joblib.dump(big_image_indexes, f"big_image_indexes.pkl")
            return big_image_indexes



    def load_image(
        self,
        case_id,
        num_imgs=config.num_images,
        img_size=config.image_size,
        rotate=0,
    ):
        data = []
        for mri_type in self.types:
            path = f"{self.folder}/{case_id}_nifti/{case_id}_{mri_type}.nii.gz"
            file_tensor = load_nii_to_tensor(path)
            print(file_tensor.shape)

            if self.is_train and self.use_indexes:
                middle = self.img_indexes[(case_id, mri_type)]
            else:
                middle = file_tensor.shape[-1] // 2
    
            num_imgs2 = num_imgs // 2
            p1 = max(0, middle - num_imgs2)
            p2 = min(file_tensor.shape[-1], middle + num_imgs2)
            file_tensor = file_tensor[:, :, p1:p2]


            padding = num_imgs - file_tensor.shape[-1]
            file_tensor = F.pad(file_tensor, (0, max(padding, 0), 0, 0, 0, 0))
            file_tensor = F.interpolate(
                file_tensor.unsqueeze(1),  
                size=(img_size, img_size),  
                mode="bilinear",  
                align_corners=False
            ).squeeze(1)
            max_abs_val = torch.abs(file_tensor).view(file_tensor.shape[0], -1).max(dim=1).values
            max_abs_val = torch.where(max_abs_val == 0, torch.tensor(1.0, device=max_abs_val.device), max_abs_val)
            file_tensor = file_tensor / max_abs_val.view(-1, 1, 1)

            data.append(file_tensor.permute(1, 2, 0).unsqueeze(0))
        # channel
        return torch.stack(data)







data = pd.read_csv("labels.csv")
train_df, tmp_df = train_test_split(data, test_size=0.3, random_state=6969)
val_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=6969)
train_dataset = BrainDataset(data=train_df, is_train=True)
print(train_dataset[0]['image'].shape)

