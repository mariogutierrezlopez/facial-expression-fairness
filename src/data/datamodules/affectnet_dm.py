# Affectnet Data Module
# Mario Gutiérrez López

# This file constains a data module implementation for affectnet to classify expressions based in the human label

import lightning as L
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from ..dataset import AffectNetDataset

from typing import Optional

class AffectNetDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 csv_train_path: str,
                 csv_test_path: str,
                 batch_size: int,
                 num_workers: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_train_path = csv_train_path
        self.csv_test_path = csv_test_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage:Optional[str]=None) -> None:

        # Tratamieto de datos para el dataset de train/val,
        #   1. Generar columna 'gender_male_bin' que contiene 1 si el sujeto es hombre y 0 si es mujer
        #   2. Eliminar archivos que no existen en el sistema
        #   3. Columna 'stratify_col' con {human_label}_{gender} para balancear los conjuntos train/val 
        #       en expresion y género
        train_df = pd.read_csv(self.csv_train_path)
        
        train_df['gender_male_bin'] = (train_df['gender_male'] > 0.5).astype(int)

        mask = train_df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x)))
        raw_df = train_df[mask].reset_index(drop=True)

        test_df = pd.read_csv(self.csv_test_path)
        test_df['gender_male_bin'] = (test_df['gender_male'] > 0.5).astype(int)
        mask = test_df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x)))
        test_df = test_df[mask].reset_index(drop=True)

        test_df, val_df = train_test_split(
            test_df,
            test_size=0.5,
            stratify=test_df['human_label'],
            random_state=42
        )

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if stage == "fit" or stage is None:
            train_balanced_df = self._apply_expression_balance_with_gender_prior(raw_df)
            # val_balanced_df = self._apply_strict_balance(val_df)
            
            self._print_contingency_table(train_balanced_df, stage_name="train")
            self._print_contingency_table(val_df, stage_name="val")
    
            self.train_ds = AffectNetDataset(self.data_dir, df=train_balanced_df, transform=transform, return_metadata=False)
            self.val_ds = AffectNetDataset(self.data_dir, df=val_df, transform=transform, return_metadata=False)
        
        if stage == "test" or stage is None:
            # test_balanced_df = self._apply_strict_balance(test_df)
            self._print_contingency_table(test_df, stage_name="test")
            self.test_ds = AffectNetDataset(self.data_dir, df=test_df, transform=transform, return_metadata=True)

    def _apply_expression_balance_with_gender_prior(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'gender_female_bin' not in df.columns:
            df['gender_female_bin'] = (df['gender_female'] > 0.5).astype(int)
        if 'gender_male_bin' not in df.columns:
            df['gender_male_bin'] = (df['gender_male'] > 0.5).astype(int)
            
        # 1. Definir el objetivo por clase (el "máximo de los mínimos")
        counts = df['human_label'].value_counts()
        global_target = counts.min()
        target_per_gender = global_target // 2
        
        print(f"Balanceo Híbrido")
        print(f"Objetivo por clase: {global_target} (Ideal: {target_per_gender} por género)")
        
        final_dfs = []
        classes = df['human_label'].unique()

        for label in classes:
            # Separar por género para esta clase específica
            women_df = df[(df['human_label'] == label) & (df['gender_female_bin'] == 1)]
            men_df = df[(df['human_label'] == label) & (df['gender_male_bin'] == 1)]
            
            n_w = len(women_df)
            n_m = len(men_df)

            # Caso A: Ambos géneros tienen suficientes fotos para el 50/50
            if n_w >= target_per_gender and n_m >= target_per_gender:
                s_women = women_df.sample(n=target_per_gender, random_state=42)
                s_men = men_df.sample(n=target_per_gender, random_state=42)
            
            # Caso B: Faltan mujeres -> Pillamos todas las mujeres y rellenamos con hombres
            elif n_w < target_per_gender:
                s_women = women_df # Todas las disponibles
                n_needed_men = global_target - n_w
                s_men = men_df.sample(n=min(n_needed_men, n_m), random_state=42)
                
            # Caso C: Faltan hombres -> Pillamos todos los hombres y rellenamos con mujeres
            else:
                s_men = men_df # Todos los disponibles
                n_needed_women = global_target - n_m
                s_women = women_df.sample(n=min(n_needed_women, n_w), random_state=42)

            final_dfs.extend([s_women, s_men])

        df_balanced = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        return df_balanced


    def _print_contingency_table(self, df:pd.DataFrame, stage_name:str="train") -> None:
        print(f"\nContingency table for {stage_name}: Strict Balance (Emotions & Gender)")
        gender_series = df['gender_male_bin'].apply(lambda x: 'Male' if x == 1 else 'Female')
        ct = pd.crosstab(df['human_label'], gender_series)
        print(ct)

        log_dir = "affectnet_logs_baseline"
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/{stage_name}_dist_balanced.csv"
        ct.to_csv(filename)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)