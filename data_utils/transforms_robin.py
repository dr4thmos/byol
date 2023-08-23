print(__name__)
from torchvision.transforms import transforms
from . import custom_transforms as ct
from astropy.stats import sigma_clip
from astropy.io import fits
import torchvision.utils as utils
import torch.nn.functional as F
import numpy as np
import torch
import os


def get_data_transforms_train_robin(**kwargs):
   
    data_transforms = transforms.Compose([
        transforms.Lambda(ct.pad_to_square),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Resize((kwargs["width"],kwargs["height"]), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms

def get_data_transforms_test_robin(**kwargs):
    data_transforms = transforms.Compose([
        transforms.Lambda(ct.pad_to_square),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Resize((kwargs["width"],kwargs["height"]), antialias=True),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms