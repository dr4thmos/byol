from pathlib import Path
import torch
import torch.utils.data
import json
import torchvision.transforms as T 
from torchvision.transforms.functional import to_pil_image

from astropy.visualization import ZScaleInterval
from torch.utils.data import Dataset
import warnings
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.io.fits.verify import VerifyWarning
import numpy as np


warnings.simplefilter('ignore', category=VerifyWarning)


CLASSES = ['background', 'spurious', 'compact', 'extended']
COLORS = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]

class RemoveNaNs(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img[np.isnan(img)] = 0
        return img
    
class ZScale(object):
    def __init__(self, contrast=0.15):
        self.contrast = contrast

    def __call__(self, img):
        interval = ZScaleInterval(contrast=self.contrast)
        min, max = interval.get_limits(img)

        img = (img - min) / (max - min)
        return img
    
class SigmaClip(object):
    def __init__(self, sigma=3, masked=True):
        self.sigma = sigma
        self.masked = masked

    def __call__(self, img):
        img = sigma_clip(img, sigma=self.sigma, masked=self.masked)
        return img
    
class MinMaxNormalize(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - img.min()) / (img.max() - img.min())
        return img
    
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32)

class Unsqueeze(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.unsqueeze(0)

class Zorro(Dataset):
    def __init__(self, data_dir, img_paths, augmentation, img_size=32):
        super().__init__()
        data_dir = Path(data_dir)
        with open(img_paths) as f:
            self.img_paths = f.read().splitlines()
        self.img_paths = [data_dir / p for p in self.img_paths]

        self.transforms = T.Compose([
                RemoveNaNs(),
                ZScale(),
                SigmaClip(),
                ToTensor(),
                torch.nn.Tanh(),
                MinMaxNormalize(),
                Unsqueeze(),
                T.Resize((img_size, img_size))
        ])

        self.augmentation = augmentation

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        img = fits.getdata(image_path)
        img = self.transforms(img)

        return self.augmentation(img)

if __name__ =='__main__':
    import sys
    sys.path.append('/home/rsortino/inaf/radio-diffusion')
    rgtrain = RGDataset('data/rg-dataset/data', 'data/rg-dataset/train_all.txt')
    batch = next(iter(rgtrain))
    image = batch
    to_pil_image(image).save('image.png')

    loader = torch.utils.data.DataLoader(rgtrain, batch_size=4, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        image = batch
        print(i, image.shape)