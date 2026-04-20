# AffWild2 Data Module
# Mario Gutiérrez López

# This file constains a data module implementation for AffWild2 to classify expressions based in the human label

import lightning as L
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from ..dataset import AffWild2Dataset

import torchvision.transforms.functional as TF

from typing import Optional

class AffWild2DatModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 csv_train_path: str,
                 csv_val_path: str,
                 batch_size: int,
                 num_workers: int,
                 test_brightness_factor: float = 1.0 #Experimentos de iluminación
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_train_path = csv_train_path
        self.csv_val_path = csv_val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_brightness_factor = test_brightness_factor

    def setup(self, stage:Optional[str]=None) -> None:

        # Tratamieto de datos para el dataset de train
        #   1. Eliminar archivos que no existen en el sistema
        train_df = pd.read_csv(self.csv_train_path)
        
        mask = train_df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.data_dir, x)))
        train_df = train_df[mask].reset_index(drop=True)

        # Tratamiento de datos para el dataset de test. El procedimiento es el mismo que en train/val
        # pero sin stratify_col
        val_df = pd.read_csv(self.csv_val_path)
        val_df = self._preprocess_metadata(val_df)
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
            self.val_ds = AffWild2Dataset(self.data_dir, df=val_df, transform=transform, return_metadata=True)
        
        if stage == "test" or stage is None:
            # test_balanced_df = self._apply_strict_balance(test_df)

            test_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Lambda(lambda img: TF.adjust_brightness(img, self.test_brightness_factor)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


            self._print_contingency_table(test_df, stage_name="test")
            self.test_ds = AffWild2Dataset(self.data_dir, df=test_df, transform=test_transforms, return_metadata=True)

    # Gender attribute to gender_male & gender_female
    def _preprocess_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'gender' in df.columns:
            clean_gender = df['gender'].astype(str).str.strip().str.lower()
            df['gender_male'] = (clean_gender == 'male').astype(int)
            df['gender_female'] = (clean_gender == 'female').astype(int)
        else:
            df['gender_male'] = -1
            df['gender_female'] = -1
            
        return df

    def _apply_subject_and_expression_balance(self, df: pd.DataFrame, max_frames_per_subject:int=200) -> pd.DataFrame:
        """
        Esta función balancea los datos para AffWild2, al ser un dataset que contiene muchas imágenes repetidas del mismo sujeto, nos aseguramos 
        de limitar ese número de imágenes mediante el parámetro `max_frames_per_subject`, balanceando el número de muestras por expresión a partir de ese
        parámetro
        """

        df = df.copy()

        # Extraer el sujeto desde la ruta (ej: 1-30-1280x720/0001.jpg -> 1-30-1280-720)
        df['real_subject'] = df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

        # Limitar frames por sujeto
        df_limited = df.groupby(['expr', 'real_subject'], group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_frames_per_subject), random_state=42),
            include_groups=True
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