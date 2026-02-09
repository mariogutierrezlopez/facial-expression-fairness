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
    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 batch_size: int,
                 num_workers: int,
                 bias_type: str,
                 bias_factor: float = 0.5,
                 target_class: int = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.bias_type = bias_type
        self.bias_factor = bias_factor
        self.target_class = target_class

    # Función obligatoria de DataModule, settea la distribución de datos
    def setup(self, stage):
        raw_df = pd.read_csv(self.csv_path)
        temps_ds = MultiPIEDataset(self.data_dir, df=raw_df)
        full_df = temps_ds.df
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

            raw_train_df = full_df[full_df['subject_id'].isin(train_subs['subject_id'])]

            train_df = self._apply_experiment_bias(raw_train_df)

            self._print_contingency_table(train_df)

            
            val_df = full_df[full_df['subject_id'].isin(val_subs['subject_id'])]
            self.train_ds = MultiPIEDataset(self.data_dir, df=train_df, transform=transform)
            self.val_ds = MultiPIEDataset(self.data_dir, df=val_df, transform=transform)
        
        if stage == "test":
            test_df = full_df[full_df['subject_id'].isin(test_subs['subject_id'])]
            self.test_ds = MultiPIEDataset(self.data_dir, df=test_df, transform=transform)

    # Esta funcion calcula el número máximo de elementos que puede haber por clase que satisfaga el ratio de género
    def _apply_experiment_bias(self, df):

        final_dfs = []
        classes = df['temp_label'].unique()


        for label in classes:
            if self.bias_type == 'stereotypical':
                if label == self.target_class:
                    target_ratio = self.bias_factor
                else:
                    target_ratio = 0.5
            else: # bias_type == 'representational'
                target_ratio = self.bias_factor
            

            available_women = df[(df['temp_label'] == label) & (df['gender'] == 1)]
            available_men = df[(df['temp_label'] == label) & (df['gender'] == 0)]

            n_women_avail = len(available_women)
            n_men_avail = len(available_men)

            if target_ratio == 0:
                n_req_women = 0
                n_req_men = n_men_avail
            elif target_ratio == 1:
                n_req_women = n_women_avail
                n_req_men = 0
            else:
                max_n_by_women = int(n_women_avail / target_ratio)
                max_n_by_men = int(n_men_avail / (1-target_ratio))

                limit_N = min(max_n_by_men, max_n_by_women)

                n_req_women = int(limit_N * target_ratio)
                n_req_men = limit_N - n_req_women
            
            #Sample data
            if n_req_women > 0:
                sampled_women = available_women.sample(n=n_req_women, random_state = 42)
            else:
                sampled_women = pd.DataFrame()
            
            if n_req_men > 0:
                sampled_men = available_men.sample(n=n_req_men, random_state = 42)
            else:
                sampled_men = pd.DataFrame()
            
            final_dfs.append(sampled_women)
            final_dfs.append(sampled_men)
        
        return pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print para ver los parámetros del experimento y la tabla con los géneros y labels
    def _print_contingency_table(self, df):
        print(f"Contingency table: {self.bias_type}, f={self.bias_factor}")
        ct = pd.crosstab(df['temp_label'], df['gender'])
        print(ct)
    
        
    # Funciones del DataModule
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)