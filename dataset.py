# ways to change datalaoder:
# instead of fixed number of slices see if model can take in variable slices
# change how fewer than expected slices are handled, currently 0 padding 
# some cv2 stuff done during dicom loading 
# in load_dicom_image pixels are normalized to pixel_size x pixel_size, maybe model can accept varying sizes
# pixel values normalized between 0 and 1
# model experiment with flattening 2d slices and then passing in all tokens from all slices, vs getting 3d patches and flattening


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

data = pd.read_csv("train_labels.csv")
train_df, val_df = train_test_split(data, test_size=0.3, random_state=6969)


device = torch.device("cuda")

def crop_img(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    c1, c2 = False, False
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
    except:
        rmin, rmax = 0, img.shape[0]
        c1 = True

    try:
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except:
        cmin, cmax = 0, img.shape[1]
        c2 = True
    bb = (rmin, rmax, cmin, cmax)
    
    if c1 and c2:
        return img[0:0, 0:0]
    else:
        return img[bb[0] : bb[1], bb[2] : bb[3]]


def extract_cropped_image_size(path):
    dicom = pydicom.dcmread(path, force=True)
    dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    try:
        data = dicom.pixel_array
    except AttributeError:
        return 0
    cropped_data = crop_img(data)
    resolution = cropped_data.shape[0] * cropped_data.shape[1]  
    return resolution
class BrainRSNADataset(Dataset):
    def __init__(
        self, data, transform=None, target="MGMT_value", mri_type="FLAIR", is_train=True, ds_type="forgot", do_load=True
    ):
        self.target = target
        self.data = data
        self.type = mri_type

        self.transform = transform
        self.is_train = is_train
        self.folder = "train" if self.is_train else "test"
        self.do_load = do_load
        self.ds_type = ds_type
        self.data['BraTS21ID'] = self.data['BraTS21ID'].astype(str).str.zfill(5)
        self.clean_data()
        self.img_indexes = self._prepare_biggest_images()

    def clean_data(self):
    
        invalid_ids = [case_id for case_id in self.data['BraTS21ID'] if not os.path.exists(f"{self.folder}/{case_id}/{self.type}")]
        self.data = self.data[~self.data['BraTS21ID'].isin(invalid_ids)]
        

            


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        row = self.data.iloc[index]

        case_id = int(row.BraTS21ID)
        target = int(row[self.target])
        _3d_images = self.load_dicom_images_3d(case_id)
        _3d_images = torch.tensor(_3d_images).float()
        if self.is_train:
            return {"image": _3d_images, "target": target, "case_id": case_id}
        else:
            return {"image": _3d_images, "case_id": case_id}

    def _prepare_biggest_images(self):
        big_image_indexes = {}
        if (f"big_image_indexes_{self.ds_type}.pkl" in os.listdir("."))\
            and (self.do_load) :
            print("Loading the best images indexes for all the cases...")
            big_image_indexes = joblib.load(f"big_image_indexes_{self.ds_type}.pkl")
            return big_image_indexes
        else:
            
            print("Calculating the best scans for every case...")
            for row in tqdm(self.data.iterrows(), total=len(self.data)):
                case_id = str(int(row[1].BraTS21ID)).zfill(5)
                 
                path = f"{self.folder}/{case_id}/{self.type}/*.dcm"
                files = sorted(
                    glob.glob(path),
                    key=lambda var: [
                        int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                    ],
                )
                
                resolutions = [extract_cropped_image_size(f) for f in files]
                if resolutions == [0] * len(resolutions):
                    middle = len(resolutions) // 2
                else:
                    middle = np.array(resolutions).argmax()
                big_image_indexes[case_id] = middle

            joblib.dump(big_image_indexes, f"big_image_indexes_{self.ds_type}.pkl")
            return big_image_indexes



    def load_dicom_images_3d(
        self,
        case_id,
        num_imgs=config.num_images,
        img_size=config.image_size,
        rotate=0,
    ):
        case_id = str(case_id).zfill(5)

        path = f"{self.folder}/{case_id}/{self.type}/*.dcm"
        files = sorted(
            glob.glob(path),
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ],
        )

        if self.is_train:
            middle = self.img_indexes[case_id]
        else:
            middle = len(files) // 2
   
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        image_stack = [load_dicom_image(f, rotate=rotate, voi_lut=True) for f in files[p1:p2]]

        img3d = np.stack(image_stack).T
        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d, n_zero), axis=-1)

        return np.expand_dims(img3d, 0)




def load_dicom_image(path, img_size=config.image_size, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path, force=True)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size))
    data = data - np.min(data)
    if np.min(data) < np.max(data):
        data = data / np.max(data)
    return data


# train_dataset = BrainRSNADataset(data=train_df, mri_type='FLAIR', ds_type=f"train")

# valid_dataset = BrainRSNADataset(data=val_df, mri_type='FLAIR', ds_type=f"train")

