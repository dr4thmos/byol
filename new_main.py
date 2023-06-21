import argparse

from data_utils.zorro import Zorro
from data_utils.robin import Robin
from data_utils.hulk import Hulk

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# Read configuration
parser = argparse.ArgumentParser(
                    prog = 'Byol main',
                    description = 'Train Byol')

# Definition
# Define dataset preprocessing
train_preprocessing      = None # if None pick standard from Dataset
validation_preprocessing = None
test_preprocessing       = None
# Define dataset augmentation
train_augmentation       = None
validation_augmentation  = None
test_augmentation        = None
# Define dataset 
train_dataset       = Hulk()
validation_dataset  = Hulk()
test_datasets       = []
# TODO add flag multilabel
test_datasets.append(Hulk())
test_datasets.append(Zorro())
test_datasets.append(Robin())

# Define dataloader
# Hyperparameters

# Initialization
# Init resnet
# Init MLP

# Train
# BYOL

# Val
# BYOL

# Test
# Classification on same dataset
# Classification on different datasets


dataloader = DataLoader()
multilabel = False
test(dataloader, multilabel)