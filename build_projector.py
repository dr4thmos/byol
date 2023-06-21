import os
import psutil
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

from data_utils.hulk import Hulk

import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from general_utils.byol_utils import _create_model_training_folder
from data_utils import transforms as my_transforms
import argparse
import yaml
import warnings
from astropy.utils.exceptions import AstropyWarning
from model_utils.resnet_base_network import ResNet


warnings.simplefilter('ignore', category=AstropyWarning)

parser = argparse.ArgumentParser(
                    prog = 'Build projections',
                    description = 'Create embeddings from images',
                    epilog = 'epilogo')

# example command line
#parser.add_argument('current_config')           # yaml in the /configs folder
parser.add_argument('-f', '--folder')           # where to log results
parser.add_argument('-e', '--experiment')       # exp name (invented, representative)

args = parser.parse_args()

#os.path.join("logs", logs_folder, exp_name)

logs_path = os.path.join("logs", args.folder, args.experiment)

config_path = os.path.join("logs", args.folder, args.experiment, "checkpoints", "config.yaml")
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

data_path = "4-HULK"
data_transform = my_transforms.get_data_transforms_eval_hulk(**config['network'])
#datalist = "labeled.json"

batch_size = 128
num_workers = 4

tobeprojected_data_labeled = Hulk(targ_dir = data_path, transform = data_transform, datalist="labeled.json")
tobeprojected_data_unlabeled = Hulk(targ_dir = data_path, transform = data_transform, datalist="unlabeled.json")

tobeprojected_loader_labeled = DataLoader(tobeprojected_data_labeled, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=False, shuffle=False)
tobeprojected_loader_unlabeled = DataLoader(tobeprojected_data_unlabeled, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=True, shuffle=False)

preview_shape = 64

limit = 8192 # px limit of sprite
n_images = int(np.floor(limit/ preview_shape))
n_images *= n_images

print(n_images)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = os.path.join(logs_path, "best_model.pt")
checkpoint = torch.load(PATH)

encoder = ResNet(**config['network'])
encoder.load_state_dict(checkpoint['online_network_state_dict'])
encoder.to(device)

resize = transforms.Resize(preview_shape)
grayscale = transforms.Grayscale()

writer = SummaryWriter(log_dir=logs_path+"-projector")

at_least_one_projection = False
for idx, data in enumerate(tobeprojected_loader_labeled):
    #torch.cuda.empty_cache()
    print("embedding creations")
    print(idx)
    batch_view = data[0].to("cuda")
    
    with torch.no_grad():
        features_batch = encoder(batch_view)

    batch_view = resize(batch_view)
    batch_view = grayscale(batch_view)
    if idx == 0:
        features = features_batch.to("cpu")
        imgs = batch_view.to("cpu")
        metadata = data[1]
        print("done")
        
    else:
        if len(features) > n_images:
            writer.add_embedding(
                features[:n_images],
                #metadata=all_labels[:2000],
                #metadata=list(zip(metadata[:4000].tolist(), list(np.zeros(4000, dtype=int)))),
                metadata=list(zip(list(np.zeros(n_images, dtype=int)), list(np.zeros(n_images, dtype=int)))),
                label_img=imgs[:n_images],
                global_step=0,
                metadata_header=['source_type', 'new_labels'],
                tag="to_idx_"+str(idx*len(batch_view))
            )
            at_least_one_projection = True
            print("-- Embedding salvati --")
            print("-- Nuova trance --")
            del features
            del imgs
            gc.collect()
            features = features_batch.to("cpu")
            imgs = batch_view.to("cpu")
        else:
            features = torch.cat((features, features_batch.to("cpu")), 0)
            imgs = torch.cat((imgs, batch_view.to("cpu")), 0)
            metadata = torch.cat((metadata, data[1]), 0)
if not(at_least_one_projection):
    writer.add_embedding(
                features,
                #metadata=all_labels[:2000],
                metadata=[tuple(x) for x in metadata.tolist()], # TODO convert to int or string
                #metadata=list(zip(metadata[:4000].tolist(), list(np.zeros(4000, dtype=int)))),
                #metadata=list(zip(list(np.zeros(len(features), dtype=int)), list(np.zeros(len(features), dtype=int)))),
                label_img=imgs,
                global_step=0,
                metadata_header=tobeprojected_data_labeled.classes.tolist(),
                #metadata_header=['source_type', 'new_labels'],
                tag="to_idx_"+str(idx*len(batch_view))
            )
    print("-- Embedding salvati --")
writer.close()