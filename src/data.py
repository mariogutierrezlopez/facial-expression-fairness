import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchvision.datasets import MNIST
import os
from torchvision import transforms
import random
from torch.utils.data import random_split, DataLoader
import torch

class FakeDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Aqui se hacel las transformaciones del dataset, 224 por resnet, 
        # cambiar los valores de normalizacion con multipie
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.1307), (0.3081))
            ])

    # 1. prepare_data() descarga las fotos
    def prepare_data(self):
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    
    # 2. `setup()` 
    def setup(self, stage: str):

        if stage == "fit":
            mnist_ful = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            
            # Calculo conjuntos train & val
            train_size = int(0.8 * len(mnist_ful))
            val_size = len(mnist_ful) - train_size

            self.mnist_train, self.mnist_val = random_split(mnist_ful, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        if stage == "test":
          self.mnist_test = MNIST(self.data_dir, download=True, train=False, transform=transforms.ToTensor())
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

