from data_utils.hulk import Hulk
from data_utils.transforms import get_data_transforms_eval_hulk
from model_utils.resnet_base_network import ResNet
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import yaml
import torch

import numpy as np
import gc



folder = "hulktest-debug"
experiment = "4-HULKhulk-3-debug-model-save"

model_name = "best_model.pt"
model_path = os.path.join(folder, experiment, model_name)
config_path = os.path.join(folder, experiment, "checkpoints/config.yaml")
#result_path = os.path.join(args.folder, args.experiment, "checkpoints/result.png")

writer = SummaryWriter(log_dir=os.path.join(folder, experiment))

print(model_path)
print(config_path)

config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

data_path = config["dataset"]

data_transform = get_data_transforms_eval_hulk(**config['network'])

eval_data = Hulk(targ_dir = data_path, transform = data_transform)
eval_data_sampled = torch.utils.data.Subset(eval_data, range(1,20000))

eval_loader = DataLoader(eval_data_sampled, batch_size=16,
                            num_workers=8, drop_last=False, shuffle=False)

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

encoder = ResNet(**config['network'])
output_feature_dim = encoder.projetion.net[0].in_features

#load pre-trained parameters

load_params = torch.load(model_path,
                         map_location=torch.device(torch.device(device)))

if 'online_network_state_dict' in load_params:
    encoder.load_state_dict(load_params['online_network_state_dict'])
    print("Parameters successfully loaded.")

# remove the projection head??
encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
encoder = encoder.to(device)

def get_features_from_encoder(encoder, loader):
    x_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            #y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    #y_train = torch.tensor(y_train)
    return x_train

encoder.eval()
x_eval = get_features_from_encoder(encoder, eval_loader)

img_size = 96

limit = 8192 # projector value
n_images = int(np.floor(limit/ img_size))
n_images *= n_images

print(n_images)

encoder.to("cuda")
for idx, data in enumerate(eval_loader):
    #torch.cuda.empty_cache()
    print("embedding creations")
    print(idx)
    batch_view = data[0].to("cuda")
    
    with torch.no_grad():
        features_batch = encoder(batch_view)

    if idx == 0:
        features = features_batch.to("cpu")
        imgs = batch_view.to("cpu")
        #metadata = data
    else:
        if len(features) > n_images:
            writer.add_embedding(
                features[:n_images],
                #metadata=all_labels[:2000],
                #metadata=list(zip(metadata[:4000].tolist(), list(np.zeros(4000, dtype=int)))),
                metadata=list(zip(list(np.zeros(n_images, dtype=int)), list(np.zeros(n_images, dtype=int)))),
                label_img=imgs[:n_images],
                global_step=777,
                metadata_header=['source_type', 'new_labels'],
                tag="to_idx_"+str(idx*len(batch_view))
            )
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
    
    del features_batch
    del batch_view
    gc.collect()
    



"""
# get the class labels for each image
class_labels = [classes[lab] for lab in labels]
"""
self.writer.close()