
from torchvision import transforms, datasets, utils
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from model_utils.resnet_base_network import ResNet
from data_utils.usecase1 import UseCase1
from data_utils.balanced_split import balanced_split
from data_utils.transforms import get_data_transforms, get_data_transforms_eval



from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torchvision
import numpy as np
import os
import torch
import sys
import yaml

import json

#num_classes = 7

experiment = 'runs/Jan25_10-12-10_1e89623a6adf/checkpoints/model.pth'
batch_size = 128

data_transform = get_data_transforms(96)
#data_transforms = torchvision.transforms.Compose([transforms.ToTensor()])

config = yaml.load(open("self_src/configs/exp1.yaml", "r"), Loader=yaml.FullLoader)

data_path = "data/croppedbackyard"

data_custom = UseCase1(targ_dir = data_path, transform = data_transform)

split = 0.2

dataset_to_be_extracted, _ = balanced_split(data_custom, split)
print("Input shape:", dataset_to_be_extracted[0][0].shape)

extr_loader = DataLoader(dataset_to_be_extracted, batch_size=1,
                          num_workers=0, drop_last=False, shuffle=False)

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
encoder = ResNet(**config['network'])
output_feature_dim = encoder.projetion.net[0].in_features

#load pre-trained parameters

load_params = torch.load(os.path.join(experiment),
                         map_location=torch.device(torch.device(device)))

if 'online_network_state_dict' in load_params:
    encoder.load_state_dict(load_params['online_network_state_dict'])
    print("Parameters successfully loaded.")

# remove the projection head
encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
encoder = encoder.to(device)

labels = {}

json_to_save = []

encoder.eval()

exp_folder_name = "experiment_cropped_backyard"
images_path = os.path.join(exp_folder_name, "images")
generated_path = os.path.join(exp_folder_name, "generated")
label_file_path = os.path.join(exp_folder_name, "labels.json")

for idx, item in enumerate(extr_loader):
    pandas_idx = extr_loader.sampler.data_source.indices[idx]
    info_row = extr_loader.sampler.data_source.dataset.info.iloc[pandas_idx]
    labels[idx] = info_row.to_dict()
    labels[idx]["original_index"] = pandas_idx
    #img = next(it)
    img = item[0]
    label = item[1]
    filename = "{}-{}-{}.png".format(pandas_idx, info_row["source_name"], info_row["survey"])
    labels[idx]["filename"] = filename
    image_file_path = os.path.join(images_path, filename)
    #generated_file_path = "{}/{}".format(generated_path, filename)

    with torch.no_grad():
        encoder.eval()
        feature_vector = encoder(img)
        #latent, pred_gen = make_predictions(model=encoder, data=img)

    utils.save_image(img, image_file_path)
    #utils.save_image(pred_gen, generated_file_path)

    #print(latent.to("cpu").numpy()[0].tolist())
    sample = torch.squeeze(feature_vector).to(device).tolist()
    json_to_save.append(sample)
    #labels["columns"].append(str(idx)+".png")

out_file = open("experiment_cropped_backyard/embeddings.json", "w+")
json.dump(json_to_save, out_file)
#out_file = open("experiment3/labels.json", "w+")
labels_pd  = pd.DataFrame.from_dict(labels, orient="index")
my_cols = set(labels_pd.columns)
my_cols.remove('original_path')
my_cols.remove('target_path')
my_cols = list(my_cols)
labels_pd_to_save = labels_pd[my_cols]
labels_pd_to_save.to_json(label_file_path, orient='index')
#json.dump(labels, out_file)

"""
img_list = []
repr_list = []
label_list = []
x_train = []
y_train = []

# get the features from the pre-trained model
for i, (x, y) in enumerate(extr_loader):
    with torch.no_grad():
        feature_vector = encoder(x)
        x_train.extend(feature_vector)
        y_train.extend(y.numpy())
        
x_train = torch.stack(x_train)
y_train = torch.tensor(y_train)
"""
"""                                 ???
if len(x_train.shape) > 2:
    x_train = torch.mean(x_train, dim=[2, 3])
    x_test = torch.mean(x_test, dim=[2, 3])
"""

#print("Data shape:", x_train.shape, y_train.shape)