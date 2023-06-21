from torchvision.transforms import transforms
#from data_utils.gaussian_blur import GaussianBlur
from astropy.stats import sigma_clip
import numpy as np
import torch
import torch.nn.functional as F

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

def norm_minmax(im):
    non_nan_idx = ~np.isnan(im)
    min_image = np.min(im[non_nan_idx])
    max_image = np.max(im[non_nan_idx])
    im[non_nan_idx] = (im[non_nan_idx] - min_image) /  (max_image - min_image)
    im[~non_nan_idx] = 0
    return im

def remove_nan(im):
    non_nan_idx = ~np.isnan(im)
    min_image = np.min(im[non_nan_idx])
    im[~non_nan_idx] = min_image
    return im



def sigma_clip_transform(t):
    if torch.rand(1) < 0.5:
        sigma = float(torch.empty(1).uniform_(float(3.0), float(7.0)).item())
        #current_min = np.min(t)
        #masked_image = sigma_clip(t[t != 0.0], sigma=sigma, maxiters=5) # non si può fare perchè restituisce un immagine con una dimensione diversa
        masked_image = sigma_clip(t, sigma=sigma, maxiters=5)
        new_max = np.max(masked_image)
        if new_max != 0:
            t /= new_max # min è 0.0
        t[masked_image.mask] = 1.0
        return t
    else:
        return t

def sigma_clip_norm(npy_image):
    if torch.rand(1) < 0.5:
        sigma = float(torch.empty(1).uniform_(float(3.0), float(7.0)).item())
        #current_min = np.min(t)
        #masked_image = sigma_clip(t[t != 0.0], sigma=sigma, maxiters=5) # non si può fare perchè restituisce un immagine con una dimensione diversa
        masked_image, lower_bound, upper_bound = sigma_clip(npy_image, sigma=sigma, maxiters=5, return_bounds=True)
        min_image = np.min(masked_image)
        max_image = np.max(masked_image)

        norm_npy_image = np.zeros(npy_image.shape)
        norm_npy_image = npy_image

        norm_npy_image = (norm_npy_image - min_image) / (max_image - min_image)
        
        norm_npy_image[masked_image.mask & (npy_image < lower_bound)] = 0.0
        norm_npy_image[masked_image.mask & (npy_image > upper_bound)] = 1.0

        return norm_npy_image
    else:
        min_image = np.min(npy_image)
        max_image = np.max(npy_image)
        npy_image = (npy_image - min_image) / (max_image - min_image)
        return npy_image

def conditioned_resize(tensor, range = [0.5, 2]):
    ### Scala le immagini più piccole di una soglia di un valore random fino all'input shape
    ### Scala le immagini più grandi dell'input shape della rete in giù
    ### Scala le immagini di mezzo a random
    ### non si capisce un cazzo ma il senso è quello
    chance = torch.rand(1)
    min_dim = min(tensor.size()[1:3])
    max_dim = max(tensor.size()[1:3])
    input_shape = 224
    
    if max_dim < input_shape:
        if max_dim >= 32:
            if chance < 0.33:
                # scale up
                new_max_dim = torch.randint(max_dim, input_shape, (1,))
                new_min_dim = int(min_dim*new_max_dim/max_dim)
            elif chance < 0.66:
                # scale down
                new_max_dim = torch.randint(int(max_dim/2), max_dim, (1,))
                new_min_dim = int(min_dim*new_max_dim/max_dim)
            else:
                # does nothing
                return tensor
        else:
            if chance < 0.5:
                # scale up
                new_max_dim = torch.randint(max_dim, input_shape, (1,))
                new_min_dim = int(min_dim*new_max_dim/max_dim)
            else:
                return tensor
        return transforms.Resize(new_min_dim)(tensor)
    else:
        if chance < 0.5:
            # Scale down
            new_max_dim = torch.randint(int(input_shape/2), input_shape, (1,))
            new_min_dim = int(min_dim*new_max_dim/max_dim)
            return transforms.Resize(new_min_dim)(tensor)
        else:
            # does nothing
            return tensor


def pad_to_square(tensor):
    H = tensor.size()[-1]
    W = tensor.size()[-2]
    if H < W:
        dif = W - H
        if dif % 2 == 0:
            up_pad = bot_pad = int(dif / 2)
        else:
            up_pad = int(dif / 2)
            bot_pad = int(dif - up_pad)
        p2d = (up_pad, bot_pad, 0, 0)
        tensor = F.pad(tensor, p2d, "constant", 0)
        
    if H > W:
        dif = H - W
        if dif % 2 == 0:
            left_pad = right_pad = int(dif / 2)
        else:
            left_pad = int(dif / 2)
            right_pad = int(dif - left_pad)
        p2d = (0, 0, left_pad, right_pad)
        tensor = F.pad(tensor, p2d, "constant", 0)
    return tensor

def pad_to_size(tensor, target_dim = 128):
    H = tensor.size()[-1]
    W = tensor.size()[-2]
    if H < W:
        dif = W - H
        if dif % 2 == 0:
            up_pad = bot_pad = int(dif / 2)
        else:
            up_pad = int(dif / 2)
            bot_pad = int(dif - up_pad)
        p2d = (up_pad, bot_pad, 0, 0)
        tensor = F.pad(tensor, p2d, "constant", 0)
        
    if H > W:
        dif = H - W
        if dif % 2 == 0:
            left_pad = right_pad = int(dif / 2)
        else:
            left_pad = int(dif / 2)
            right_pad = int(dif - left_pad)
        p2d = (0, 0, left_pad, right_pad)
        tensor = F.pad(tensor, p2d, "constant", 0)
    
    H2 = tensor.size()[-1]
    W2 = tensor.size()[-2]

    if H2 < target_dim:
        dif = target_dim - H2
        if dif % 2 == 0:
            up_pad = bot_pad = int(dif / 2)
        else:
            up_pad = int(dif / 2)
            bot_pad = int(dif - up_pad)
    else:
        up_pad = bot_pad = 0
    if W2 < target_dim:
        dif = target_dim - W2
        if dif % 2 == 0:
            left_pad = right_pad = int(dif / 2)
        else:
            left_pad = int(dif / 2)
            right_pad = int(dif - left_pad)
    else:
        left_pad = right_pad = 0

    p2d = (up_pad, bot_pad, left_pad, right_pad)
    """
    if F.pad(tensor, p2d, "constant", 0).shape != torch.Size([1, 96, 96]):
        print(F.pad(tensor, p2d, "constant", 0).shape)
    """
    return F.pad(tensor, p2d, "constant", 0)

def shift_and_pad_to_size(tensor, target_dim = 224):
    H = tensor.size()[-1]
    W = tensor.size()[-2]

    if H < W:
        dif = W - H
        up_pad = np.random.randint(0, dif, size=1)[0]
        bot_pad = dif - up_pad
        p2d = (up_pad, bot_pad, 0, 0)
        tensor = F.pad(tensor, p2d, "constant", 0)
        
    if H > W:
        dif = H - W
        left_pad = np.random.randint(0, dif, size=1)[0]
        right_pad = dif - left_pad
        p2d = (0, 0, left_pad, right_pad)
        tensor = F.pad(tensor, p2d, "constant", 0)
    
    H2 = tensor.size()[-1]
    W2 = tensor.size()[-2]

    if H2 < target_dim:
        dif = target_dim - H2
        up_pad = np.random.randint(0, dif, size=1)[0]
        bot_pad = dif - up_pad
    else:
        up_pad = bot_pad = 0
    if W2 < target_dim:
        dif = target_dim - W2
        left_pad = np.random.randint(0, dif, size=1)[0]
        right_pad = dif - left_pad
    else:
        left_pad = right_pad = 0

    p2d = (up_pad, bot_pad, left_pad, right_pad)
    """
    if F.pad(tensor, p2d, "constant", 0).shape != torch.Size([1, 96, 96]):
        print(F.pad(tensor, p2d, "constant", 0).shape)
    """
    """
    if ret.shape != torch.Size([1, 96, 96]):
        print(tensor.shape)
        print(ret.shape)
        print("---")
    """
    return F.pad(tensor, p2d, "constant", 0)

def resize(tensor, target_dim = 96):
    H = tensor.size()[-1]
    W = tensor.size()[-2]
    if H < target_dim:
        dif = target_dim - H
        if dif % 2 == 0:
            up_pad = bot_pad = dif / 2
        else:
            up_pad = int(dif / 2)
            bot_pad = dif - up_pad
    if W < target_dim:
        dif = target_dim - H
        if dif % 2 == 0:
            left_pad = right_pad = dif / 2
        else:
            left_pad = int(dif / 2)
            right_pad = dif - up_pad

    p2d = (up_pad, bot_pad, left_pad, right_pad)

    return F.pad(tensor, p2d, "constant", 0)
