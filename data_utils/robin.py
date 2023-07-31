import os
import pathlib
import torch
import pandas as pd
from astropy.io import fits
#from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Tuple, Dict, List
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T 

from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clip
# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

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



class Robin(Dataset):
    
    def __init__(self, targ_dir: str = "2-ROBIN", transform=None, datalist="info.json") -> None:
        #self.paths = list(pathlib.Path(targ_dir).glob("*.npy"))
        self.targ_dir = targ_dir
        self.preprocessing = T.Compose([
                RemoveNaNs(),
                ZScale(),
                SigmaClip(),
                ToTensor(),
                torch.nn.Tanh(),
                MinMaxNormalize(),
                Unsqueeze(),
        ])
        
        self.transform  = transform
        self.info       = self.load_info()
        self.class_to_idx = self.enumerate_classes()
        self.num_classes = len(self.class_to_idx)
        self.weights = self.setup_weights()
        print(self.info.describe())

    def setup_weights(self):
        label_to_count = self.info["source_type"].value_counts()
        weights =  1.0 / label_to_count[self.info["source_type"]]
        return torch.DoubleTensor(weights.to_list())

    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
    
    def load_info(self, datalist = 'info.json'):
        #info_file = os.path.join(self.targ_dir, 'info.json')
        info_file = os.path.join(self.targ_dir, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    # Deve essere nella stessa precisione dei pesi del modello
    def load_image(self, index: int) -> np.float32:
        "Opens an image via a path and returns it."
        try:
            image_path = os.path.join(self.targ_dir, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(index)
        return self.preprocessing(img), self.info.iloc[index]["source_type"]
    
    def filter_dataset(self):
        # probably a Sampler will be used
        pass

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.info)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform:
            #return self.transform(img), class_idx # return data, label (X, y)
            return self.transform(img), self.class_to_idx[label] # return data, label (X)
        else:
            #return img, class_idx # return data, label (X, y)
            return img, self.class_to_idx[label] # return data, label (X)



if __name__ == "__main__":
    #import sys
    #sys.path.append('/home/rsortino/inaf/radio-diffusion')
    
    transforms = transforms.Compose([transforms.ToTensor(),
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([128,128]),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    dataset = Robin('2-ROBIN', transforms)
    batch = next(iter(dataset))
    image = batch[0]
    to_pil_image(image).save('image.png')

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        image = batch[0]
        print(i, image.shape)