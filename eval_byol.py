import os
import random
import numpy as np

import matplotlib.pyplot as plt

import json

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from data_utils.hulk import Hulk

from torch.utils.data.dataloader import DataLoader
from general_utils.losses import FocalLoss

from data_utils import transforms as my_transforms
import argparse
import yaml
import warnings
from astropy.utils.exceptions import AstropyWarning
from model_utils.resnet_base_network import ResNet
from model_utils.linear_classifier import LinearClassifier
from model_utils.multi_layer_classifier import MultiLayerClassifier

# optimizer = SGD lr=0.03, momentum=0.9
# 10 epochs
# classes ['COMPACT', 'DIFFUSE', 'DIFFUSE-LARGE', 'EXTENDED', 'FILAMENT', 'RADIO-GALAXY', 'RING', 'ARTEFACT']

# BCELoss
# Imagenet weights freezed + linear classificator
# end weighted f1-score 70.43
# Random weights freezed + linear classificator
# Random weights + linear classificator
# Byol wrights freezed + linear classificator

# FocalLoss
# Imagenet weights freezed + linear classificator
# Random weights freezed + linear classificator
# Random weights + linear classificator
# Byol wrights freezed + linear classificator

warnings.simplefilter('ignore', category=AstropyWarning)

parser = argparse.ArgumentParser(
                    prog = 'Eval model on multilabel data',
                    description = 'Provide model metrics',
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
datalist = "labeled.json"

batch_size = 256
num_workers = 4

data_labeled = Hulk(targ_dir = data_path, transform = data_transform, datalist="labeled.json")

loader_labeled = DataLoader(data_labeled, batch_size=batch_size,
                                  num_workers=num_workers, drop_last=False, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = os.path.join(logs_path, "best_model.pt")
checkpoint = torch.load(PATH)

#config['network'][]

config['network']["pretrained_weights"] = True
encoder = ResNet(**config['network'])

#encoder.load_state_dict(checkpoint['online_network_state_dict'])
encoder.to(device)

multilabel_linear_classificator = LinearClassifier(encoder.repr_shape, data_labeled.num_classes).to(device)
#multilabel_linear_classificator = MultiLayerClassifier(encoder.repr_shape, data_labeled.num_classes).to(device)
"""
multilabel_linear_classificator = torch.nn.Sequential(
        torch.nn.Linear(encoder.repr_shape, data_labeled.num_classes),
        torch.nn.Sigmoid()
        ).to(device)
"""
optimizer_classificator = torch.optim.SGD(list(multilabel_linear_classificator.parameters()), lr=0.03, momentum=0.9)

multilabel_classification_loss = nn.BCELoss()#nn.BCEWithLogitsLoss() ##FocalLoss(gamma=2.) #nn.BCEWithLogitsLoss() #nn.BCELoss()

running_loss = 0.0
running_accuracy = 0.0

#pred_arr_stacked = []
#original_arr_stacked = []

for epoch in range(30):
    print("Val-Classification Iteration")
    print(epoch)
    for idx, data in enumerate(loader_labeled):
        batch_view = data[0].to(device)
        gtruth = data[1].to(device)
        with torch.no_grad():
            features = encoder(batch_view, repr=True)
    
        predicted = multilabel_linear_classificator(features)
        ## DEBUG HERE
        loss_class = multilabel_classification_loss(predicted, gtruth)
        #loss_class = sigmoid_focal_loss(predicted, gtruth)
        #print(loss_class.item())
        loss_class.backward()
        #running_loss += loss_class.item()
        optimizer_classificator.step()
        optimizer_classificator.zero_grad()

        pred_arr = np.round(predicted.float().cpu().detach().numpy())
        original_arr = gtruth.float().cpu().detach().numpy()
        if idx == 0:
            pred_arr_concatenated = pred_arr
            original_arr_concatenated = original_arr
        else:
            pred_arr_concatenated = np.concatenate((pred_arr_concatenated, pred_arr), axis=0)
            original_arr_concatenated = np.concatenate((original_arr_concatenated, original_arr), axis=0)

    
    #mcm = multilabel_confusion_matrix(original_arr_concatenated, pred_arr_concatenated, labels = data_labeled.classes)
    
    
    min = random.randint(0, len(original_arr_concatenated)-15)
    max = min + 15
    print(original_arr_concatenated[min:max])
    print(pred_arr_concatenated[min:max])
    
    mcm = multilabel_confusion_matrix(original_arr_concatenated, pred_arr_concatenated)
    print(mcm)

    f, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    for i in range(data_labeled.num_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix(original_arr_concatenated[:, i],
                                                    pred_arr_concatenated[:, i], normalize="all")*100,
                                    display_labels=[0, i])
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(data_labeled.classes[i])
        if i<8:
            disp.ax_.set_xlabel('')
        if i%4!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    if (epoch == 0) or (epoch == 29):
        plt.savefig('conf_mat_imagenet{}.png'.format(epoch))
    plt.close()

    #print(len(pred_arr_concatenated))
    class_rep = classification_report(original_arr_concatenated, pred_arr_concatenated, target_names=data_labeled.classes, output_dict=False, zero_division=0.) # TODO
    print(class_rep)
    class_rep = classification_report(original_arr_concatenated, pred_arr_concatenated, target_names=data_labeled.classes, output_dict=True, zero_division=0.) # TODO
    
    if (epoch == 0) or (epoch == 29):
        with open('report_imagenet_{}.json'.format(epoch), 'w') as outfile:
            json.dump(class_rep, outfile)
    
    weighted_f1_score = class_rep["weighted avg"]["f1-score"]*100
    print(weighted_f1_score)