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
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
def download(x):
    id = "11E7B5-UYEhVbnjofqJauQDwASYq9g-ES"
    file = pathlib.Path("best_model_276E_96I.pth")
    if file.exists ():
        pass
    else:
        gdown.download(id=id, output='best_model_276E_96I.pth', quiet=False)
    return 

