#   dataset.py
#   Mario Gutiérrez López

import torch
from torchvision.datasets import VisionDataset
import pandas as pd
from PIL import Image
import os

class MultiPIEDataset(VisionDataset):

    def __init__(self, root, df, transform=None, target_transform = None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        # print(f"El archivo {root} existe: {os.path.exists(root)}")
        # print(f"Datagrame: {df}")
        # print(f"Columnas del df{df.columns}")

        # Eliminar las filas que no tienen emoción
        self.df = df.copy().reset_index(drop=True)

        if self.df.empty:
            self.df = pd.DataFrame(columns=['rel_path', 'temp_label', 'gender', 'ethnicity', 'age', 'camera_id'])
            self.labels = []
            return
        
        if 'temp_label' not in self.df.columns:
            self.df['temp_label'] = self._generate_labels()

        if 'temp_label' in self.df.columns:
            self.df = self.df[self.df["temp_label"] != -1].reset_index(drop=True)
        
        if not self.df.empty:
            mask = self.df['rel_path'].apply(lambda x: os.path.exists(os.path.join(self.root, x)))
            self.df = self.df[mask].reset_index(drop=True)

        if 'temp_label' not in self.df.columns:
                self.df['temp_label'] = self._generate_labels()


        self.df = self.df[self.df["temp_label"] != -1].reset_index(drop=True)

        # Eliminar filas donde la imagen no existe
        mask = self.df['rel_path'].apply(lambda x: os.path.exists(os.path.join(self.root, x)))
        self.df = self.df[mask].reset_index(drop=True)

        # Añadir el label
        self.labels = self.df["temp_label"].tolist()




    def __len__(self):
        return len(self.df)
    
    # Self method to create labels column based on session_id and recording_id
    def _generate_labels(self):
        def map_row(row):
            s = str(row['session_id'])
            r = row['recording_id']

            if r == 1: return 0 # Neutral
            if 'session01' in s and r == 2: return 1 # Smile
            if 'session02' in s and r == 2: return 2 # Surprise
            if 'session02' in s and r == 3: return 3 # Squint
            if 'session03' in s and r == 2: return 1 # Smile
            if 'session03' in s and r == 3: return 4 # Disgust
            if 'session04' in s and r == 2: return 5 # Scream
            if 'session04' in s and r == 3: return 0 # Neutral
            return -1

        return self.df.apply(map_row, axis=1).tolist()
    


    def __getitem__(self, index):
        row = self.df.iloc[index]

        target = self.labels[index]

        img_path = os.path.join(self.root, row["rel_path"])

        image = Image.open(img_path).convert("RGB")


        demographics = {
            'gender': row['gender'],
            'ethnicity': row['ethnicity'],
            'age': row['age'],
            'camera_id': row['camera_id']
        }

        # Transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return {
            'image': image,
            'target': target,
            'meta': demographics
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
                'video': row['video'],
                'frame_idx': row['frame_idx'],
                'subject': row['subject'],
                'split': row['split'],
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