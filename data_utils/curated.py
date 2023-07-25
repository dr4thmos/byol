import os
import pathlib
import torch
import pandas as pd
from astropy.io import fits
#from PIL import Image
import numpy as np
from torch.utils.data import Dataset
#from torchvision import transforms
from typing import Tuple, Dict, List
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

from sklearn.preprocessing import MultiLabelBinarizer

class Curated(Dataset):
    """
    Dataset of extended sources with a border around region of 2.5*size?
    Source is in the middle
    Classes are known = only interesting sources
    """
    
    def preprocess():
        pass
        #return transforms

    def __init__(self, targ_dir: str = "cutout_factor2.5/meerkat", transform=None, datalist='info.json') -> None:
        #self.paths = list(pathlib.Path(targ_dir).glob("*.npy"))
        self.targ_dir       = targ_dir
        self.transform      = transform
        self.info           = self.load_info(datalist)

        """
        self.labels = ['UNKNOWN', 'ARTEFACT', 'BACKGROUND', 'BORDER', 'COMPACT', 'DIFFUSE', 'DIFFUSE-LARGE',
        'EXTENDED', 'FILAMENT', 'MOSAICING', 'RADIO-GALAXY', 'RING', 'WTF']
        """
        self.labels = ['UNKNOWN']
        #self.weights = [0.5, 1., 1., 1., 1., 1., 1.]
        self.classes = self.multilabel_one_hot_encoding()
        #self.class_to_idx   = self.enumerate_classes()
        self.num_classes    = len(self.classes)
        #self.weights        = self.setup_weights()
        print(self.info.describe())


    def setup_weights(self):
        label_to_count = self.info["source_type"].value_counts()
        weights =  1.0 / label_to_count[self.info["source_type"]]
        return torch.DoubleTensor(weights.to_list())

    def multilabel_one_hot_encoding(self):
        mlb = MultiLabelBinarizer(classes=self.labels)
        self.info["one_hot"] = mlb.fit_transform(self.info["source_type"]).tolist()
        
        print(mlb.classes_)
        return mlb.classes_
        

    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
    
    def load_info(self, datalist):
        info_file = os.path.join(self.targ_dir, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    # Deve essere nella stessa precisione dei pesi del modello
    def load_image(self, index: int) -> np.float32:
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.targ_dir, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
            #img = self.transform(img)
            #return img, mask
        except:
            print(index)
        return img, torch.FloatTensor(self.info.iloc[index]["one_hot"])
    
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
            return self.transform(img), label # return data, label (X)
        else:
            #return img, class_idx # return data, label (X, y)
            return img, label # return data, label (X)

if __name__ == "__main__":
    #import sys
    #sys.path.append('/home/rsortino/inaf/radio-diffusion')
    
    transforms = transforms.Compose([transforms.ToTensor(),
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([128,128]),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
    dataset = Hulk('2-ROBIN', transforms)
    batch = next(iter(dataset))
    image = batch[0]
    to_pil_image(image).save('image.png')

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        image = batch[0]
        print(i, image.shape)