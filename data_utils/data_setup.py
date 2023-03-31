"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
#from data_utils.usecase1 import UseCase1
from data_utils.usecase1_with_labels import UseCase1
from data_utils.balanced_split import balanced_split

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    data_path: str,
    split: float,
    batch_size: int,
    transforms = None,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

  data_custom = UseCase1(targ_dir = data_path, transform = transforms)

  #train_dataset, test_dataset = torch.utils.data.random_split(data_custom, [train_size, test_size])
  train_dataset, test_dataset = balanced_split(data_custom, split)

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
      sampler=WeightedRandomSampler(train_dataset.dataset.weights[train_dataset.indices], len(train_dataset.dataset.weights[train_dataset.indices]))
  )
  test_dataloader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
      sampler=WeightedRandomSampler(test_dataset.dataset.weights[test_dataset.indices], len(test_dataset.dataset.weights[test_dataset.indices]))
  )

  return train_dataloader, test_dataloader


def create_dataloaders_prediction(
    data_path: str,
    split: float,
    batch_size: int = 1,
    transforms = None,
    num_workers: int=1
):

  data_custom = UseCase1(targ_dir = data_path, transform = transforms)

  sample, _ = balanced_split(data_custom, split)

  # Turn images into data loaders
  dataloader = DataLoader(
      sample,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return dataloader
