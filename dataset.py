import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=4):
      super().__init__()
      self.data_dir = data_dir
      self.batch_size = batch_size
      self.num_workers = num_workers

    # Just DOwnload the data, If already exist we must load them here!
    def prepare_data(self):
      # single GPU
      # Usually we use a Custome Dataset to import data here
      datasets.MNIST(self.data_dir, train=True, download=True) # Just to download the TRAINING data
      datasets.MNIST(self.data_dir, train=False, download=True) # Just to download the TEST data


    def setup(self, stage):
      # multi GPU
      entire_dataset = datasets.MNIST( # Loads the TRAINING data only (and keep some for VALIDATION)
          root=self.data_dir,
          train=True,
          transform=transforms.Compose([ #
              transforms.ToTensor(),
              # transforms.RandomHorizontalFlip(), # Ruin the images
              # transforms.RandomVerticalFlip() # Ruin the images
          ]),
          download=False,
      )
      self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
      self.test_ds = datasets.MNIST( # Loads the TEST data only
          root=self.data_dir,
          train=False,
          transform=transforms.ToTensor(),
          download=False,
      )

    def train_dataloader(self):
        return DataLoader(
          self.train_ds,
          batch_size=self.batch_size,
          num_workers=self.num_workers,
          persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
          self.val_ds,
          batch_size=self.batch_size,
          num_workers=self.num_workers,
          shuffle=False,
          persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
          self.test_ds,
          batch_size=self.batch_size,
          num_workers=self.num_workers,
          shuffle=False,
          persistent_workers=True
        )
