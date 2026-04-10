#   dataset.py
#   Mario Gutiérrez López

import torch
from torchvision.datasets import VisionDataset
import pandas as pd
from PIL import Image
import os
from src.utils.utils import generate_labels
class MultiPIEDataset(VisionDataset):

    def __init__(self, root, df, transform=None, target_transform = None, return_metadata=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root


        # Este parámetro maneja la información que se devuelve por cada objeto en el __getitem__()
        #   train/val -> False | Solo se devuelve la imagen y la clase objetivo
        #   train/val -> True | Además de la imagen y el target, se devuelven los metadatos (edad, género, raza, iluminación...)
        self.return_metadata = return_metadata


        processed_df = generate_labels(df)

        if not processed_df.empty:
            mask = processed_df['abs_path'].apply(
                lambda x: os.path.exists(
                    str(x).replace('data_cropped', 'data').replace('/multiview/', '/')
                )
            )
            self.df = processed_df[mask].reset_index(drop=True)
        else:
            self.df = processed_df
        
        if not self.df.empty:
            self.labels = self.df["temp_label"].tolist()
        else:
            self.labels = []

    def __len__(self):
        return len(self.df)   


    def __getitem__(self, index):
        row = self.df.iloc[index]

        target = self.labels[index]

        img_path = str(row["abs_path"]).replace('data_cropped', 'data').replace('/multiview/', '/')

        image = Image.open(img_path).convert("RGB")


        demographics = {
            'gender': row['gender'],
            'gender_male': 1 if row['gender'] == 'Male' else 0,
            'gender_female': 1 if row['gender'] == 'Female' else 0,
            'ethnicity': row['ethnicity'],
            'age': row['age'],
            'camera_id': row['camera_id']
        }

        # Transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.return_metadata:
            return {
                'image': image,
                'target': target,
                'meta': demographics
            }
        else:
            return{
                'image': image,
                'target': target,
            }

class AffectNetDataset(VisionDataset):
    
    def __init__(self, root, df, transform=None, target_transform = None, return_metadata=False):
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Este parámetro maneja la información que se devuelve por cada objeto en el __getitem__()
        #   train/val -> False | Solo se devuelve la imagen y la clase objetivo
        #   train/val -> True | Además de la imagen y el target, se devuelven los metadatos (edad, género, raza, iluminación...)
        self.return_metadata = return_metadata

        self.df = df.copy().reset_index(drop=True)

        mask = self.df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.root, x)))
        self.df = self.df[mask].reset_index(drop=True)


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        target = row['human_label']

        img_path = os.path.join(self.root, row['image_path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Lógica para devolver metadatos:
        if self.return_metadata:
            demographics = {
                'age': row['age'],
                'gender_male': row['gender_male'],
                'gender_female': row['gender_female'],
                'yaw': row['yaw'],
                'pitch': row['pitch'],
                'roll': row['roll'],
                'race_white': row['race_white'],
                'race_black': row['race_black'],
                'race_asian': row['race_asian'],
                'race_indian': row['race_indian'],
                'race_middle_eastern': row['race_middle_eastern'],
                'race_latino_hispanic': row['race_latino_hispanic'],
                'hf_ratio': row['hf_ratio'],
                'log_var': row['log_var'],
                'quality_score': row['quality_score'],
                'quality_bin': row['quality_bin'],
                'illumination': row['illumination']
            }

            return {
                'image': image,
                'target': target,
                'meta': demographics
            }
        else:
            return {
                'image': image,
                'target': target
            }

class AffWild2Dataset(VisionDataset):

    def __init__(self, root, df, transform=None, target_transform = None, return_metadata=False):
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Este parámetro maneja la información que se devuelve por cada objeto en el __getitem__()
        #   train/val -> False | Solo se devuelve la imagen y la clase objetivo
        #   train/val -> True | Además de la imagen y el target, se devuelven los metadatos (edad, género, raza, iluminación...)
        self.return_metadata = return_metadata

        self.df = df.copy().reset_index(drop=True)

        # Eliminar filas donde no se sepa la expresion
        self.df.dropna(subset=['expr'])

        # Verificación de que las imágenes existen (comentado por rendimiento)
        # mask = self.df['image_path'].apply(lambda x: os.path.exists(os.path.join(self.root, x)))
        # self.df = self.df[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        target = int(row['expr'])

        img_path = os.path.join(self.root, row['image_path'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Lógica para devolver metadatos:
        if self.return_metadata:
            demographics = {
                # 'video': row['video'],
                # 'frame_idx': row['frame_idx'],
                # 'subject': row['subject'],
                # 'split': row['split'],
                # 'yaw': row['yaw'],
                # 'pitch': row['pitch'],
                # 'roll': row['roll'],
                # 'ethnicity': row['ethnicity'],
                # 'gender': row['gender'],
                # 'age': row['age'],
                'illumination': row['illumination']
            }

            return {
                'image': image,
                'target': target,
                'meta': demographics
            }
        else:
            return {
                'image': image,
                'target': target
            }