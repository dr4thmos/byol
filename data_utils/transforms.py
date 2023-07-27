print(__name__)
from torchvision.transforms import transforms
import torchvision.utils as utils
#import sys
#sys.path.append('byol/data_utils/custom_transforms')
from . import custom_transforms as ct
#import custom_transforms as ct
from astropy.stats import sigma_clip
import numpy as np
import torch
import torch.nn.functional as F
import os
from astropy.io import fits

def get_data_transforms_zorro(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([
        #transforms.Lambda(sigma_clip_norm),
        #transforms.Lambda(ct.add_norm_channel),
        #transforms.Lambda(norm_minmax),
        #transforms.ToTensor(),
        
        #transforms.RandomEqualize(p= 1.0),
        #transforms.ColorJitter(brightness = (0.8, 1), contrast = (0.9, 1.0), saturation = 0.0, hue = 0.0),
        #transforms.RandomGrayscale(p=0.3),
        #transforms.Lambda(conditioned_resize),
        #transforms.Lambda(shift_and_pad_to_size),
        #transforms.RandomResizedCrop(size=kwargs["width"], scale=(0.4, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([kwargs["width"],kwargs["width"]]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomInvert(),
        #transforms.Grayscale(num_output_channels = 3),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
        #transforms.RandomApply([color_jitter], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
    ])
    return data_transforms

def get_data_transforms_zorro_eval(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([
        #transforms.Lambda(sigma_clip_norm),
        #transforms.Lambda(ct.add_norm_channel),
        #transforms.Lambda(norm_minmax),
        #transforms.ToTensor(),
        
        #transforms.RandomEqualize(p= 1.0),
        #transforms.ColorJitter(brightness = (0.8, 1), contrast = (0.9, 1.0), saturation = 0.0, hue = 0.0),
        #transforms.RandomGrayscale(p=0.3),
        #transforms.Lambda(conditioned_resize),
        #transforms.Lambda(shift_and_pad_to_size),
        #transforms.RandomResizedCrop(size=kwargs["width"], scale=(0.4, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize(kwargs["width"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomInvert(),
        transforms.ConvertImageDtype(dtype=torch.float32)
        #transforms.RandomApply([color_jitter], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
    ])
    return data_transforms

def get_data_transforms(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.

    input_shape = (
        kwargs["input_shape"]["width"],
        kwargs["input_shape"]["height"],
        kwargs["input_shape"]["channels"]
    )
    
    data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          #transforms.Lambda(sigma_clip_norm),
                                          #transforms.Lambda(add_norm_channel),
                                          transforms.ToTensor(),
                                          #transforms.RandomEqualize(p= 1.0),
                                          transforms.RandomInvert(),
                                          transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
                                          #transforms.Lambda(conditioned_resize),
                                          #transforms.Lambda(shift_and_pad_to_size),
                                          transforms.Resize(input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ConvertImageDtype(dtype=torch.float32)
                                          #transforms.RandomApply([color_jitter], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2),
                                          #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          ])
    return data_transforms

def get_data_transforms_hulk(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([
        #transforms.Lambda(sigma_clip_norm),
        transforms.Lambda(ct.add_norm_channel),
        #transforms.Lambda(norm_minmax),
        transforms.ToTensor(),
        
        #transforms.RandomEqualize(p= 1.0),
        transforms.ColorJitter(brightness = (0.8, 1), contrast = (0.9, 1.0), saturation = (0.5, 1.0), hue = 0.0),
        transforms.RandomGrayscale(p=0.3),
        #transforms.Lambda(conditioned_resize),
        #transforms.Lambda(shift_and_pad_to_size),
        transforms.RandomResizedCrop(size=kwargs["width"], scale=(0.4, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize(kwargs["width"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomInvert(),
        transforms.ConvertImageDtype(dtype=torch.float32)
        #transforms.RandomApply([color_jitter], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
    ])
    return data_transforms

def get_data_transforms_curated(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([
        #transforms.Lambda(sigma_clip_norm),
        transforms.Lambda(ct.add_norm_channel),
        #transforms.Lambda(norm_minmax),
        transforms.ToTensor(),
        
        #transforms.RandomEqualize(p= 1.0),
        transforms.ColorJitter(brightness = (0.8, 1), contrast = (0.9, 1.0), saturation = (0.5, 1.0), hue = 0.0),
        transforms.RandomGrayscale(p=0.3),
        #transforms.Lambda(conditioned_resize),
        #transforms.Lambda(shift_and_pad_to_size),
        transforms.RandomResizedCrop(size=kwargs["width"], scale=(0.4, 1.0), ratio=(1.0, 1.0), antialias=True),
        #transforms.RandomRotation(degrees=15.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
        transforms.Resize(kwargs["width"], antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomInvert(),
        transforms.ConvertImageDtype(dtype=torch.float32)
        #transforms.RandomApply([color_jitter], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
    ])
    return data_transforms

def get_data_transforms_eval_curated(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    """
    input_shape = (
        kwargs["input_shape"]["width"],
        kwargs["input_shape"]["height"],
        kwargs["input_shape"]["channels"]
    )
    """
    #preview_shape = kwargs["preview_shape"]
    data_transforms = transforms.Compose([transforms.Lambda(ct.add_norm_channel),
                                        transforms.ToTensor(),
                                        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.Resize(kwargs["input_shape"]["width"]),
                                        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms

def get_data_transforms_hulk_without_normalization(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([
        #transforms.Lambda(sigma_clip_norm),
        #transforms.Lambda(ct.add_norm_channel),
        #transforms.Lambda(norm_minmax),
        transforms.Lambda(ct.remove_nan),
        
        transforms.ToTensor(),
        
        #transforms.RandomEqualize(p= 1.0),
        #transforms.ColorJitter(brightness = (0.8, 1), contrast = (0.9, 1.0), saturation = (0.5, 1.0), hue = 0.0),
        #transforms.RandomGrayscale(p=0.3),
        #transforms.Lambda(conditioned_resize),
        #transforms.Lambda(shift_and_pad_to_size),
        transforms.RandomResizedCrop(size=kwargs["width"], scale=(0.4, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize(kwargs["width"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.RandomInvert(),
        transforms.ConvertImageDtype(dtype=torch.float32)
        #transforms.RandomApply([color_jitter], p=0.8),
        #transforms.RandomGrayscale(p=0.2),
        #GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
    ])
    return data_transforms

def get_data_transforms_eval_hulk(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    """
    input_shape = (
        kwargs["input_shape"]["width"],
        kwargs["input_shape"]["height"],
        kwargs["input_shape"]["channels"]
    )
    """
    #preview_shape = kwargs["preview_shape"]
    data_transforms = transforms.Compose([transforms.Lambda(ct.add_norm_channel),
                                        transforms.ToTensor(),
                                        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.Resize(kwargs["input_shape"]["width"]),
                                        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms

def get_data_transforms_eval(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    #color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    input_shape = 96

    data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          #transforms.Lambda(sigma_clip_norm),
                                          transforms.Lambda(ct.histeq),
                                          transforms.ToTensor(),
                                          transforms.Lambda(ct.pad_to_size),
                                          transforms.Resize(input_shape)
                                          ])
    return data_transforms

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

if __name__ == "__main__":
    image_path = os.path.join("G002.5+0.0IFx_Mosaic_Mom0_cutout0001483.fits")
    img = fits.getdata(image_path).astype(np.float32)
    transform = transforms.Lambda(ct.add_norm_channel)
    
    img = transform(img)
    tensorize = transforms.ToTensor()
    img = tensorize(img)
    utils.save_image(img[0,:,:], "G002.5+0.0IFx_Mosaic_Mom0_cutout0001483_0.png")
    utils.save_image(img[1,:,:], "G002.5+0.0IFx_Mosaic_Mom0_cutout0001483_1.png")
    utils.save_image(img[2,:,:], "G002.5+0.0IFx_Mosaic_Mom0_cutout0001483_2.png")
    utils.save_image(img, "G002.5+0.0IFx_Mosaic_Mom0_cutout0001483.png")
