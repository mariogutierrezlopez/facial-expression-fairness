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

    def setup(self, stage=None):

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
            train_balanced_df = self._apply_expression_balance(train_df)
            # val_balanced_df = self._apply_strict_balance(val_df)
            
            self._print_contingency_table(train_balanced_df, stage_name="train")
            self._print_contingency_table(val_df, stage_name="val")
    
            self.train_ds = AffWild2Dataset(self.data_dir, df=train_balanced_df, transform=transform, return_metadata=False)
            self.val_ds = AffWild2Dataset(self.data_dir, df=val_df, transform=transform, return_metadata=False)
        
        if stage == "test" or stage is None:
            # test_balanced_df = self._apply_strict_balance(test_df)
            self._print_contingency_table(test_df, stage_name="test")
            self.test_ds = AffWild2Dataset(self.data_dir, df=test_df, transform=transform, return_metadata=True)

    # Funcion de balanceo 50/50 en género
    #TODO Verificar si esta función aplica a AffWild2
    # def _apply_strict_balance(self, df):

    #     df = df.copy()
    #     df['gender_male_bin'] = (df['gender_male'] > 0.5).astype(int)
    #     df['gender_female_bin'] = (df['gender_female'] > 0.5).astype(int)
        
    #     final_dfs = []
    #     classes = df['expr'].unique()

    #     for label in classes:
    #         available_women = df[(df['expr'] == label) & (df['gender_female_bin'] == 1)]
    #         available_men = df[(df['expr'] == label) & (df['gender_male_bin'] == 1)]

    #         n_limit = min(len(available_women), len(available_men))

    #         if n_limit > 0:
    #             sampled_women = available_women.sample(n=n_limit, random_state=42)
    #             sampled_men = available_men.sample(n=n_limit, random_state=42)
    #             final_dfs.extend([sampled_women, sampled_men])
    #         else:
    #             print(f" la clase {label} ha sido excluida por falta de representantes de un género.")

    #     # Mezclar el dataset final
    #     return pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # TODO Verificar si esta funcion aplica a AffWild2
    # def _apply_expression_balance(self, df):
    #     df = df.copy()
        
    #     # 1. Encontrar el "Máximo de los Mínimos"
    #     # Contamos cuántas imágenes hay por cada emoción y nos quedamos con la cifra más baja
    #     counts = df['expr'].value_counts()
    #     n_limit = counts.min()
        
    #     print(f"--- Balanceando dataset ---")
    #     print(f"Mínimo común encontrado: {n_limit} muestras por clase.")
        
    #     final_dfs = []
    #     classes = df['expr'].unique()

    #     # 2. Samplear n_limit para cada clase
    #     for label in classes:
    #         # Filtramos por emoción y sacamos exactamente n_limit muestras
    #         sampled_class = df[df['expr'] == label].sample(n=n_limit, random_state=42)
    #         final_dfs.append(sampled_class)

    #     # 3. Concatenar, barajar y resetear índice
    #     df_balanced = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
    #     print(f"Dataset final balanceado: {len(df_balanced)} imágenes totales.")
    #     return df_balanced
    

    # # TODO Revisar esta funcion (en AffWild2 no existe el género)
    # def _apply_expression_balance_with_gender_prior(self, df):
    #     df = df.copy()

    #     if 'gender_female_bin' not in df.columns:
    #         df['gender_female_bin'] = (df['gender_female'] > 0.5).astype(int)
    #     if 'gender_male_bin' not in df.columns:
    #         df['gender_male_bin'] = (df['gender_male'] > 0.5).astype(int)
            
    #     # 1. Definir el objetivo por clase (el "máximo de los mínimos")
    #     counts = df['expr'].value_counts()
    #     global_target = counts.min()
    #     target_per_gender = global_target // 2
        
    #     print(f"Balanceo Híbrido")
    #     print(f"Objetivo por clase: {global_target} (Ideal: {target_per_gender} por género)")
        
    #     final_dfs = []
    #     classes = df['expr'].unique()

    #     for label in classes:
    #         # Separar por género para esta clase específica
    #         women_df = df[(df['expr'] == label) & (df['gender_female_bin'] == 1)]
    #         men_df = df[(df['expr'] == label) & (df['gender_male_bin'] == 1)]
            
    #         n_w = len(women_df)
    #         n_m = len(men_df)

    #         # Caso A: Ambos géneros tienen suficientes fotos para el 50/50
    #         if n_w >= target_per_gender and n_m >= target_per_gender:
    #             s_women = women_df.sample(n=target_per_gender, random_state=42)
    #             s_men = men_df.sample(n=target_per_gender, random_state=42)
            
    #         # Caso B: Faltan mujeres -> Pillamos todas las mujeres y rellenamos con hombres
    #         elif n_w < target_per_gender:
    #             s_women = women_df # Todas las disponibles
    #             n_needed_men = global_target - n_w
    #             s_men = men_df.sample(n=min(n_needed_men, n_m), random_state=42)
                
    #         # Caso C: Faltan hombres -> Pillamos todos los hombres y rellenamos con mujeres
    #         else:
    #             s_men = men_df # Todos los disponibles
    #             n_needed_women = global_target - n_m
    #             s_women = women_df.sample(n=min(n_needed_women, n_w), random_state=42)

    #         final_dfs.extend([s_women, s_men])

    #     df_balanced = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    #     return df_balanced

    def _apply_expression_balance(self, df):
        df = df.copy()
        
        # 1. Ignorar clases que no existen de verdad en el dataframe
        counts = df['expr'].value_counts()
        counts = counts[counts > 0] # Nos quedamos solo con lo que existe
        
        # 2. Definir un objetivo. 
        # Si counts.min() es muy pequeño (ej. 10 fotos), el dataset será enano.
        # Mejor pon un mínimo fijo, por ejemplo 2000 muestras por clase.
        target_n = max(2000, counts.min()) 
        
        print(f"--- Balanceando AffWild2 ---")
        print(f"Muestras por clase objetivo: {target_n}")
        
        final_dfs = []
        classes = counts.index.tolist()

        for label in classes:
            sub_df = df[df['expr'] == label]
            n_available = len(sub_df)
            
            # Si tiene menos de lo que queremos, hacemos oversampling (replace=True)
            replace = n_available < target_n
            
            sampled_class = sub_df.sample(n=target_n, replace=replace, random_state=42)
            final_dfs.append(sampled_class)

        df_balanced = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Dataset final: {len(df_balanced)} imágenes.")
        return df_balanced
    
    def _print_contingency_table(self, df, stage_name="train"):
        print(f"\n--- Distribucion for {stage_name} ---")
        
        counts = df['expr'].value_counts().sort_index()
        print(counts)
        # Lo guardamos en CSV por si quieres graficarlo luego
        log_dir = "affwild2_logs_baseline"
        os.makedirs(log_dir, exist_ok=True)
        filename = f"{log_dir}/{stage_name}_dist_expression.csv"
        counts.to_csv(filename)
        
        print(f"Total samples in {stage_name}: {len(df)}")
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)