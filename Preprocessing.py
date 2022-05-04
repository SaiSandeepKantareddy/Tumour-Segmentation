import torch
import numpy as np
import pickle
import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import matplotlib.image as img
import cv2
import pydicom as dicom
from skimage.transform import resize
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import gdown
import warnings
import argparse
import pathlib
import nibabel as nib
warnings.filterwarnings("ignore")

# class CustomDataset(Dataset):
#     def __init__(self, image_paths, target_paths, train=True,tfms=None):   # initial logic       happens like transform
#         self.image_paths = image_paths
#         self.target_paths = target_paths
#         self.transforms = transforms.ToTensor()
#         self.tfms=tfms
    
#     def FillHole(self,mask):
#         contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         len_contour = len (contours)
#         contour_list = []
#         for i in range(len_contour):
#             drawing = np.zeros_like(mask, np.uint8)  # create a black image
#             img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
#             contour_list.append(img_contour)

#         out = sum(contour_list)
#         return out
    
#     def __getitem__(self, index):
#         #print(index)
#         image = self.image_paths[index]
#         img = np.array(255.0 / np.amax(image) * image, dtype = np.uint8)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#         cl1 = clahe.apply(img)
#         t_image=cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
#         mask = self.target_paths[index]
#         try:
#             if type(self.FillHole(mask.astype('uint8')))==np.ndarray:
#                 mask=self.FillHole(mask.astype('uint8'))
#             else:
#                 mask=mask.astype('uint8')
#         except:
#             mask=mask.astype('uint8')
#         t_image = self.transforms(t_image)
#         mask =self.transforms(mask)
#         return t_image, mask

#     def __len__(self):  # return count of sample we have

#         return len(self.image_paths)

class CustomDataset(Dataset):
    def __init__(self, image_paths, train=True,tfms=None):   # initial logic       happens like transform
        self.image_paths = image_paths
        self.transforms = transforms.ToTensor()
        self.tfms=tfms
    
    def FillHole(self,mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len (contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
            contour_list.append(img_contour)

        out = sum(contour_list)
        return out
    
    def __getitem__(self, index):
        #print(index)
        image = self.image_paths[index]
        img = np.array(255.0 / np.amax(image) * image, dtype = np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        t_image=cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
        t_image = self.transforms(t_image)
        return t_image

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)
    
def download():
    id = "11E7B5-UYEhVbnjofqJauQDwASYq9g-ES"
    file = pathlib.Path("best_model_276E_96I.pth")
    if file.exists ():
        pass
    else:
        gdown.download(id=id, output='best_model_276E_96I.pth', quiet=False)
    best_model = torch.load('./best_model_276E_96I.pth')
    return best_model
    
def load_nifti_img(filepath):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data())
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta 

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def extract_data_(s1):
    out_nii_array, meta = load_nifti_img(s1)
    h=list(range(out_nii_array.shape[2]))
    for i in range(out_nii_array.shape[2]):
        out_nii_array[0:,0:,i]=window_image(out_nii_array[0:,0:,i],3177,6355)
    return out_nii_array,h



def extract_data(s1,s2):
    out_nii_array, meta = load_nifti_img(s1)
    out_nii_roi, meta = load_nifti_img(s2)
    if out_nii_array.shape[-1]==out_nii_roi.shape[-1]:
        h=glob.glob('/'.join(s1.split('/')[0:8])+'/'+"/*/*/*/IM*",recursive=True)
        for i in range(out_nii_array.shape[2]):
            out_nii_array[0:,0:,i]=window_image(out_nii_array[0:,0:,i],3177,6355)
        return out_nii_array,out_nii_roi,h
    else:
        print('Your Input and ROI images are not matching')
        
def IOU(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    union=(np.sum(fsum+ssum))-intersect
    union=intersect/union
    union=np.mean(union)
    union=round(union,3)
    return union

def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice 