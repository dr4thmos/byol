import os
import pathlib
import torch
import pandas as pd

#from PIL import Image
import numpy as np
from torch.utils.data import Dataset
#from torchvision import transforms
from typing import Tuple, Dict, List

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

class UseCase1(Dataset):
    
    def __init__(self, targ_dir: str, transform=None) -> None:
        #self.paths = list(pathlib.Path(targ_dir).glob("*.npy"))
        self.targ_dir = targ_dir
        self.transform  = transform
        self.info       = self.load_info()
        self.class_to_idx = self.enumerate_classes()
        self.weights = self.setup_weights()


    def setup_weights(self):
        label_to_count = self.info["source_type"].value_counts()
        weights =  1.0 / label_to_count[self.info["source_type"]]
        return torch.DoubleTensor(weights.to_list())

    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
    
    def load_info(self):
        #info_file = os.path.join(self.targ_dir, 'info.json')
        info_file = os.path.join(self.targ_dir, 'info.json')
        df = pd.read_json(info_file, orient="index")
        return df

    # Deve essere nella stessa precisione dei pesi del modello
    def load_image(self, index: int) -> np.float32:
        "Opens an image via a path and returns it."
        try:
            image_path = os.path.join(self.targ_dir, self.info.iloc[index]["target_path"])
        except:
            print(index)
        return np.load(image_path).astype(np.float32), self.info.iloc[index]["source_type"]
    
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
    import sys
    #sys.path.append('/home/rsortino/inaf/radio-diffusion')
    rgtrain = UseCase1('2-ROBIN', )
    batch = next(iter(rgtrain))
    image = batch
    to_pil_image(image).save('image.png')

    loader = torch.utils.data.DataLoader(rgtrain, batch_size=4, shuffle=False, num_workers=0)
    for i, batch in enumerate(loader):
        image = batch
        print(i, image.shape)