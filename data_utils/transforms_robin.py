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


def get_data_transforms_eval_robin(**kwargs):
    input_shape = (
        kwargs["input_shape"]["width"],
        kwargs["input_shape"]["height"],
        kwargs["input_shape"]["channels"]
    )
    
    data_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(size=eval(input_shape)[0]),
        #transforms.Lambda(sigma_clip_norm),
        transforms.Lambda(ct.add_norm_channel),
        transforms.ToTensor(),
        #transforms.RandomEqualize(p= 1.0),
        #transforms.RandomInvert(),
        #transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        #transforms.Lambda(ct.conditioned_resize),
        transforms.Lambda(ct.pad_to_square),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((kwargs["width"],kwargs["height"])),
        #transforms.Resize(input_shape[0]),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ConvertImageDtype(dtype=torch.float32)
        #transforms.RandomApply([color_jitter], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
    ])
    return data_transforms

def get_data_transforms_test_robin(**kwargs):
    
    data_transforms = transforms.Compose([
        #transforms.Lambda(ct.add_norm_channel),
        #transforms.ToTensor(),
        transforms.Lambda(ct.pad_to_square),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Resize((kwargs["width"],kwargs["height"]), antialias=True),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms