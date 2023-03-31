from torchvision import transforms, datasets
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from model_utils.resnet_base_network import ResNet
from data_utils.usecase1 import UseCase1
from data_utils.balanced_split import balanced_split, quotas_balanced_split
from data_utils.transforms import get_data_transforms_eval, get_data_transforms

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
import argparse


#experiment_folder = "runs"
#experiment_folder = "benchmark"

#experiments = sorted(os.listdir(experiment_folder))
#experiment = experiments[-1]
#experiment = "1_channel_sigma3_minmax_histeq"
#print(experiment)

parser = argparse.ArgumentParser(
                    prog = 'Byol Trainer',
                    description = 'Train Byol',
                    epilog = 'epilogo')

#parser.add_argument('current_config')          # positional argument
parser.add_argument('-e', '--experiment')      # option that takes a value
parser.add_argument('-f', '--folder')      # option that takes a value
parser.add_argument('-epc', '--epochs', default=10)
parser.add_argument('-cl', '--num_classes', default=10)
parser.add_argument('-bs', '--batch_size', default=128)
#parser.add_argument('-v', '--verbose',
#                    action='store_true')  # on/off flag

args = parser.parse_args()

model_path = os.path.join(args.folder, args.experiment, "checkpoints/model.pth")
config_path = os.path.join(args.folder, args.experiment, "checkpoints/config.yaml")
result_path = os.path.join(args.folder, args.experiment, "checkpoints/result.png")

print(model_path)
print(config_path)

config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

data_transform = get_data_transforms(**config['network'])

data_path = os.path.join("data", config["dataset"])

data_custom = UseCase1(targ_dir = data_path, transform = data_transform)

split = 0.5

train_dataset, test_dataset = quotas_balanced_split(data_custom, split)
print("Input shape:", train_dataset[0][0].shape)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
encoder = ResNet(**config['network'])
output_feature_dim = encoder.projetion.net[0].in_features

#load pre-trained parameters

load_params = torch.load(model_path,
                         map_location=torch.device(torch.device(device)))

if 'online_network_state_dict' in load_params:
    encoder.load_state_dict(load_params['online_network_state_dict'])
    print("Parameters successfully loaded.")

# remove the projection head
encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
encoder = encoder.to(device)

def get_features_from_encoder(encoder, loader):
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(x)
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

            
    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train

encoder.eval()
x_train, y_train = get_features_from_encoder(encoder, train_loader)
x_test, y_test = get_features_from_encoder(encoder, test_loader)

if len(x_train.shape) > 2:
    x_train = torch.mean(x_train, dim=[2, 3])
    x_test = torch.mean(x_test, dim=[2, 3])
    
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader


scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)


train_loader, test_loader = create_data_loaders_from_arrays(torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        #self.sigmoid = torch.nn.Sigmoid(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

logreg = LogisticRegression(output_feature_dim, args.num_classes)
logreg = logreg.to(device)

optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

for epoch in range(int(args.epochs)):
#     train_acc = []
    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()        
        
        logits = logreg(x)
        
        predictions = torch.argmax(logits, dim=1)
        #print(torch.unique(predictions))
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
    
    total = 0
    if epoch % eval_every_n_epochs == 0:
        y_pred = []
        y_true = []
        correct = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = logreg(x)
            predictions = torch.argmax(logits, dim=1)
            
            total += y.size(0)
            correct += (predictions == y).sum().item()

            y_pred.extend(predictions) 
            y_true.extend(y)
            
        acc = 100 * correct / total
        print(f"Testing accuracy: {np.mean(acc)}")

classes = ('C1', 'C2', 'C3', 'EXT', 'EXT-MI', 'DIFFUSE', 'BG')

cf_matrix = confusion_matrix(y_true, y_pred, labels=[i for i, _ in enumerate(classes)], normalize="true")
"""
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *100, index = [i for i in classes],
                    columns = [i for i in classes])
"""
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                    columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig(result_path)
"""
for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    repr = encoder.forward(x)

"""