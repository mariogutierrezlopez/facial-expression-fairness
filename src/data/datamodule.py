import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torchvision.datasets import MNIST
import os
from torchvision import transforms
import random
from torch.utils.data import random_split, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import MultiPIEDataset

class MultiPIEDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, csv_path: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        full_df = pd.read_csv(self.csv_path)

        # Obtener sujetos únicos y género para estratificar por persona
        subjects_df = full_df[['subject_id', 'gender']].drop_duplicates()


        # División de datos en 70% train, 15% val, 15% test
        train_subs, temp_subs = train_test_split(
            subjects_df,
            test_size=0.3,
            stratify=subjects_df['gender'],
            random_state=42
        )

        val_subs, test_subs = train_test_split(
            temp_subs,
            test_size=0.5,
            stratify=temp_subs['gender'],
            random_state=42
        )

        # Transform MultiPIE -> Resnet50
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Asignación en dataframes
        if stage == "fit":
            train_df = full_df[full_df['subject_id'].isin(train_subs['subject_id'])]
            val_df = full_df[full_df['subject_id'].isin(val_subs['subject_id'])]

            self.train_ds = MultiPIEDataset(self.data_dir, df=train_df, transform=transform)
            self.val_ds = MultiPIEDataset(self.data_dir, df=val_df, transform=transform)
        
        if stage == "test":
            test_df = full_df[full_df['subject_id'].isin(test_subs['subject_id'])]
            self.test_ds = MultiPIEDataset(self.data_dir, df=test_df, transform=transform)

    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)