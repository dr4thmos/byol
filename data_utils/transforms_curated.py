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

def get_data_transforms_curated(**kwargs):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    data_transforms = transforms.Compose([
        transforms.Lambda(ct.add_norm_channel),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness = (0.8, 1), contrast = (0.9, 1.0), saturation = (0.5, 1.0), hue = 0.0),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomResizedCrop(size=kwargs["width"], scale=(0.4, 1.0), ratio=(1.0, 1.0), antialias=True),
        transforms.Resize(kwargs["width"], antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms

def get_data_transforms_eval_curated(**kwargs):
    data_transforms = transforms.Compose([transforms.Lambda(ct.add_norm_channel),
                                        transforms.ToTensor(),
                                        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.Resize(kwargs["input_shape"]["width"]),
                                        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    return data_transforms

def random_norm_3_channel(im):
    # Speculari con parametri diversi?
    # Quindi immagine 1 traformata ad esempio con log 0.1 e la seconda con 10.0
    # Quindi immagine 1 traformata ad esempio con log 0.1 e la seconda con 10.0
    # Quindi immagine 1 traformata ad esempio con log 0.1 e la seconda con 10.0
    # 3 canali con 3 normalizzazioni casuali?
    # Qui servirebbe fare uno studio sulle normalizzazioni più safe
    # Prenderei log 0.1 e log 10
    # Quello di renato
    return im
    

def add_norm_channel(im,nbr_bins=1024):
    # obtain the image histogram
    non_nan_idx = ~np.isnan(im)
    min_image = np.min(im[non_nan_idx])
    im[~non_nan_idx] = min_image # set NaN values to min_values
    imhist,bins = np.histogram(im.flatten(),nbr_bins, density=True)
    # derive the cumulative distribution function, CDF
    cdf = imhist.cumsum()      
    # normalise the CDF
    cdf = cdf / cdf[-1]
    
    im2 = np.interp(im.flatten(),bins[:-1],cdf).reshape(im.shape)

    sigma = 3.0
    #current_min = np.min(t)
    #masked_image = sigma_clip(t[t != 0.0], sigma=sigma, maxiters=5) # non si può fare perchè restituisce un immagine con una dimensione diversa
    masked_image, lower_bound, upper_bound = sigma_clip(im, sigma=sigma, maxiters=5, return_bounds=True)
    min_image = np.min(masked_image)
    max_image = np.max(masked_image)

    norm_npy_image = np.zeros(im.shape)
    norm_npy_image = im

    norm_npy_image = (norm_npy_image - min_image) / (max_image - min_image)
    
    norm_npy_image[masked_image.mask & (im < lower_bound)] = 0.0
    norm_npy_image[masked_image.mask & (im > upper_bound)] = 1.0

    # use linear interpolation of CDF to find new pixel values
    min_image = np.min(im)
    max_image = np.max(im)
    im = (im - min_image) /  (max_image - min_image)
    return np.dstack((im, norm_npy_image, im2))

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
