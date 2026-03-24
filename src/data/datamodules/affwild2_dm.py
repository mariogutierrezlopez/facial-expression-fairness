# AffWild2 Data Module
# Mario Gutiérrez López

# This file constains a data module implementation for AffWild2 to classify expressions based in the human label

import lightning as L
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from ..dataset import AffWild2Dataset # Corregir esto

from typing import Optional

class AffWild2DatModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 csv_train_path: str,
                 csv_val_path: str,
                 batch_size: int,
                 num_workers: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_train_path = csv_train_path
        self.csv_val_path = csv_val_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage:Optional[str]=None) -> None:

        # Tratamieto de datos para el dataset de train
        #   1. Eliminar archivos que no existen en el sistema
        train_df = pd.read_csv(self.csv_train_path)
        
        mask = train_df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x)))
        train_df = train_df[mask].reset_index(drop=True)

        # Tratamiento de datos para el dataset de test. El procedimiento es el mismo que en train/val
        # pero sin stratify_col
        val_df = pd.read_csv(self.csv_val_path)
        mask = val_df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x)))
        val_df = val_df[mask].reset_index(drop=True)

        # A partir del csv de val, generar dataframes validation y test
        val_df, test_df = train_test_split(
            val_df,
            test_size=0.5,
            stratify=val_df['expr'],
            random_state=42
        )

        # Resize a 224 para ResNet50 y normalizar con pesos ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if stage == "fit" or stage is None:
            train_balanced_df = self._apply_subject_and_expression_balance(train_df, max_frames_per_subject=250)
            # val_balanced_df = self._apply_strict_balance(val_df)
            
            self._print_contingency_table(train_balanced_df, stage_name="train")
            self._print_contingency_table(val_df, stage_name="val")
    
            self.train_ds = AffWild2Dataset(self.data_dir, df=train_balanced_df, transform=transform, return_metadata=False)
            self.val_ds = AffWild2Dataset(self.data_dir, df=val_df, transform=transform, return_metadata=False)
        
        if stage == "test" or stage is None:
            # test_balanced_df = self._apply_strict_balance(test_df)
            self._print_contingency_table(test_df, stage_name="test")
            self.test_ds = AffWild2Dataset(self.data_dir, df=test_df, transform=transform, return_metadata=True)

    def _apply_subject_and_expression_balance(self, df: pd.DataFrame, max_frames_per_subject:int=200) -> pd.DataFrame:
        """
        Esta función balancea los datos para AffWild2, al ser un dataset que contiene muchas imágenes repetidas del mismo sujeto, nos aseguramos 
        de limitar ese número de imágenes mediante el parámetro `max_frames_per_subject`, balanceando el número de muestras por expresión a partir de ese
        parámetro
        """

        df = df.copy()

        df_limited = df.groupby(['expr', 'subject'], group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_frames_per_subject), random_state=42)
        ).reset_index(drop=True)

        # Calcular el mínimo de muestras por emoción para limitar el dataset a ese valor `target_n``
        counts = df_limited['expr'].value_counts()
        target_n = counts.min()

        # Muestrear todas las clases a ese valor
        final_dfs = []
        for label in counts.index:
            sub_df = df_limited[df_limited['expr'] == label]

            sampled_class = sub_df.sample(n=target_n, replace=False, random_state=42)
            final_dfs.append(sampled_class)
        
        # Mezclar
        df_balanced = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        return df_balanced




    def _apply_expression_balance(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Ignorar clases que no existen de verdad en el dataframe
        counts = df['expr'].value_counts()
        counts = counts[counts > 0] # Nos quedamos solo con lo que existe
        
        # Definir un objetivo. 
        target_n = max(2000, counts.min()) 
        
        print(f"--- Balanceando AffWild2 ---")
        print(f"Muestras por clase objetivo: {target_n}")
        
        final_dfs = []
        classes = counts.index.tolist()

        for label in classes:
            sub_df = df[df['expr'] == label]
            n_available = len(sub_df)
            
            replace = n_available < target_n
            
            sampled_class = sub_df.sample(n=target_n, replace=replace, random_state=42)
            final_dfs.append(sampled_class)

        df_balanced = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Dataset final: {len(df_balanced)} imágenes.")
        return df_balanced
    
    def _print_contingency_table(self, df:pd.DataFrame, stage_name:str="train") -> None:
        print(f"\n--- Distribucion for {stage_name} ---")
        
        counts = df['expr'].value_counts().sort_index()
        print(counts)
        # Lo guardamos en CSV por si quieres graficarlo luego
        log_dir = "affwild2_logs_baseline"
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/{stage_name}_dist_expression.csv"
        counts.to_csv(filename)
        
        print(f"Total samples in {stage_name}: {len(df)}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)