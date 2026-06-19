# MultiPIE Data Module
# Mario Gutiérrez López

# This file constains a data module implementation for MultiPIE to classify expressions
# The main feature is stereotipcial and representational bias based in Dominguez-Catena et al.

import lightning as L
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from ..dataset import MultiPIEDataset

POSE_BINS = {
    "frontal": ["14_0", "05_1", "05_0"],
    "profile": ["12_0", "09_0", "11_0"],
}

# ILLUMINATION_BINS = {
#     # Luces directas / Flash activo
#     "illuminated": ["01", "02", "03", "06", "07", "08", "09", "15", "16", "17"], 
    
#     # Luz ambiental / Sombras / Foco trasero o apagado
#     "dark": ["00", "04", "05", "10", "11", "12", "13", "14", "18", "19"]         
# }


ILLUMINATION_BINS = {
    # Luces directas / Flash activo
    "illuminated": ["06", "07"], 
    
    # Luz ambiental / Sombras / Foco trasero o apagado
    "dark": ["00", "19"]         
}

class MultiPIEDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 csv_path: str,
                 batch_size: int,
                 num_workers: int,
                 bias_type: str,
                 bias_factor: float = 0.5,
                 n_limit:int = 0,
                 target_class: int | None = None,
                 pose_scenario: str = "H_Frontal_M_Profile",
                 brightness_scenario: str = "H_Illuminated_M_Dark"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.bias_type = bias_type
        self.bias_factor = bias_factor
        self.target_class = target_class

        self.pose_scenario = pose_scenario
        self.brightness_scenario = brightness_scenario


    # Función obligatoria de DataModule, settea la distribución de datos
    def setup(self, stage) -> None:
        full_df = pd.read_csv(self.csv_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
        if stage == "fit" or stage is None:
            #TRAIN
            raw_train_df = full_df[full_df['split']=='train'].copy()

            # Conseguir un subconjunto divisible entre 4
            def enforce_divisibility(group):
                target_n = (len(group) // 4) * 4 # Si hay 37, 37//4 = 9. 9*4 = 36.
                if target_n == 0:
                    return pd.DataFrame(columns=group.columns)
                return group.sample(n=target_n, random_state=42)
                
            raw_train_df = raw_train_df.groupby(
                ['temp_label', 'gender', 'camera_id', 'illumination_id'], 
                group_keys=False
            ).apply(enforce_divisibility).reset_index(drop=True)


            raw_train_df = self._add_pose_column(raw_train_df)
            raw_train_df = self._add_brightness_column(raw_train_df)
            train_df = self._apply_experiment_bias(raw_train_df)


            self._print_contingency_table(train_df, stage_name="train")
            self.train_ds = MultiPIEDataset(self.data_dir, df=train_df, transform=transform, return_metadata=True)


            #VAL
            raw_val_df = full_df[full_df['split'] == 'val'].copy()
            raw_val_df = raw_val_df.groupby(
                ['temp_label', 'gender', 'camera_id', 'illumination_id'], 
                group_keys=False
            ).apply(enforce_divisibility).reset_index(drop=True)


            raw_val_df = self._add_pose_column(raw_val_df)
            raw_val_df = self._add_brightness_column(raw_val_df)
            val_df = self._apply_experiment_bias(raw_val_df)

            
            self._print_contingency_table(val_df, stage_name="val")
            self.val_ds = MultiPIEDataset(self.data_dir, df=val_df, transform=transform, return_metadata=True)
        
        if stage == "test" or stage is None:
            test_df = full_df[full_df['split']=='test'].copy()

            test_df = self._add_pose_column(test_df)
            test_df = self._add_brightness_column(test_df)
            
            self._print_contingency_table(test_df, stage_name="test")
            self.test_ds = MultiPIEDataset(self.data_dir, df=test_df, transform=transform, return_metadata=True)


    # FUNCION PARA OBTENER DATASET BALANCEADO CON LA CLASE MAS BAJA
    def _apply_experiment_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        final_dfs = []
        classes = df['temp_label'].unique()

        # Función auxiliar para reducir el volumen a la mitad (50%) 
        # manteniendo la equidad en la distribución de cámaras e iluminación.
        def sample_half(group_df):
            target_n = int(len(group_df) * 0.5)
            if target_n == 0:
                return pd.DataFrame(columns=group_df.columns)
            return group_df.sample(n=target_n, random_state=42)

        for label in classes:

            # ---------------------------------------------------------
            # SESGO ESTEREOTÍPICO POR ILUMINACIÓN (Solo target_class)
            # ---------------------------------------------------------
            if self.bias_type == "stereotypical_brightness":
                if label == self.target_class:
                    if self.brightness_scenario == "H_Illuminated_M_Dark":
                        m_ill = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['brightness'] == "illuminated")]
                        w_dark  = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['brightness'] == "dark")]
                        final_dfs.extend([m_ill, w_dark])
                        
                    elif self.brightness_scenario == "M_Illuminated_H_Dark":
                        w_ill = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['brightness'] == "illuminated")]
                        m_dark  = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['brightness'] == "dark")]
                        final_dfs.extend([w_ill, m_dark])
                else:
                    other_df = df[df['temp_label'] == label]
                    sampled_other = other_df.groupby(
                        ['gender', 'camera_id', 'illumination_id'], 
                        group_keys=False
                    ).apply(sample_half).reset_index(drop=True)
                    final_dfs.append(sampled_other)

            # ---------------------------------------------------------
            # SESGO REPRESENTACIONAL POR ILUMINACIÓN (Todas las clases)
            # ---------------------------------------------------------
            elif self.bias_type == "brightness":
                if self.brightness_scenario == "H_Illuminated_M_Dark":
                    m_ill = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['brightness'] == "illuminated")]
                    w_dark  = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['brightness'] == "dark")]
                    final_dfs.extend([m_ill, w_dark])
                    
                elif self.brightness_scenario == "M_Illuminated_H_Dark":
                    w_ill = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['brightness'] == "illuminated")]
                    m_dark  = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['brightness'] == "dark")]
                    final_dfs.extend([w_ill, m_dark])
            
            # ---------------------------------------------------------
            # SESGO ESTEREOTÍPICO POR POSE (Solo en target_class)
            # ---------------------------------------------------------
            elif self.bias_type == "stereotypical_pose": # <--- CORREGIDO AQUÍ (ahora es elif)
                
                # A. CLASE OBJETIVO
                if label == self.target_class:
                    if self.pose_scenario == "H_Frontal_M_Profile":
                        m_front = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['pose'] == "frontal")]
                        w_prof  = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['pose'] == "profile")]
                        final_dfs.extend([m_front, w_prof])
                        
                    elif self.pose_scenario == "M_Frontal_H_Profile":
                        w_front = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['pose'] == "frontal")]
                        m_prof  = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['pose'] == "profile")]
                        final_dfs.extend([w_front, m_prof])
                        
                    elif self.pose_scenario == "Balanced_Half":
                        target_df = df[df['temp_label'] == label]
                        sampled_target = target_df.groupby(
                            ['gender', 'camera_id', 'illumination_id'], 
                            group_keys=False
                        ).apply(sample_half).reset_index(drop=True)
                        final_dfs.append(sampled_target)
                
                # B. RESTO DE CLASES
                else:
                    other_df = df[df['temp_label'] == label]
                    sampled_other = other_df.groupby(
                        ['gender', 'camera_id', 'illumination_id'], 
                        group_keys=False
                    ).apply(sample_half).reset_index(drop=True)
                    final_dfs.append(sampled_other)

            # ---------------------------------------------------------
            # EXPERIMENTO 3 ORIGINAL (DESBALANCEO DE POSE/CÁMARA GLOBAL)
            # ---------------------------------------------------------
            elif self.bias_type == "pose":
                if self.pose_scenario == "H_Frontal_M_Profile":
                    m_front = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['pose'] == "frontal")]
                    w_prof  = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['pose'] == "profile")]
                    final_dfs.extend([m_front, w_prof])
                    
                elif self.pose_scenario == "M_Frontal_H_Profile":
                    w_front = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['pose'] == "frontal")]
                    m_prof  = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['pose'] == "profile")]
                    final_dfs.extend([w_front, m_prof])
                    
                elif self.pose_scenario == "H_Frontal_M_Frontal":
                    m_front = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['pose'] == "frontal")]
                    w_front = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['pose'] == "frontal")]
                    final_dfs.extend([m_front, w_front])
                    
                elif self.pose_scenario == "H_Profile_M_Profile":
                    m_prof  = df[(df['temp_label'] == label) & (df['gender'] == "Male") & (df['pose'] == "profile")]
                    w_prof  = df[(df['temp_label'] == label) & (df['gender'] == "Female") & (df['pose'] == "profile")]
                    final_dfs.extend([m_prof, w_prof])
            
            # ---------------------------------------------------------
            # EXPERIMENTOS 1 y 2 ORIGINALES (SESGO REPRESENTACIONAL y ESTEREOTÍPICO POR GÉNERO)
            # ---------------------------------------------------------
            else:
                if self.bias_type == "stereotipical":
                    current_f = self.bias_factor if label == self.target_class else 0.5
                else: 
                    current_f = self.bias_factor
                
                available_women = df[(df['temp_label'] == label) & (df['gender'] == "Female")]
                available_men = df[(df['temp_label'] == label) & (df['gender'] == "Male")]
                
                def sample_group(group_df, ratio):
                    target_n = int(len(group_df) * ratio)
                    if target_n == 0:
                        return pd.DataFrame(columns=group_df.columns)
                    return group_df.sample(n=target_n, random_state=42)

                sampled_women = available_women.groupby(['camera_id', 'illumination_id'], group_keys=False).apply(lambda g: sample_group(g, current_f)).reset_index(drop=True)
                sampled_men = available_men.groupby(['camera_id', 'illumination_id'], group_keys=False).apply(lambda g: sample_group(g, 1.0 - current_f)).reset_index(drop=True)

                final_dfs.extend([sampled_women, sampled_men])

        return pd.concat(final_dfs, ignore_index=True)

    def _add_pose_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asigna la clase 'frontal' o 'profile' segun la camara y filtra el resto """
        df = df.copy()

        def map_pose(cam):
            cam_str = str(cam)
            if cam_str in POSE_BINS["frontal"]: return "frontal"
            if cam_str in POSE_BINS["profile"]: return "profile"
            return "other"

        df['pose'] = df['camera_id'].apply(map_pose)

        return df[df['pose'] != "other"] 
    
    def _add_brightness_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def map_brightness(ill):
            ill_str = str(ill).zfill(2) # Ajusta el zfill si tus IDs en el CSV no tienen el '0' a la izquierda (ej. "1" vs "01")
            if ill_str in ILLUMINATION_BINS["illuminated"]: return "illuminated"
            if ill_str in ILLUMINATION_BINS["dark"]: return "dark"
            return "other"

        df['brightness'] = df['illumination_id'].apply(map_brightness)
        return df[df['brightness'] != "other"]
    

    # Print para ver los parámetros del experimento y la tabla con los géneros y labels
    def _print_contingency_table(self, df, stage_name="train") -> None:
        scenario = self.brightness_scenario if "brightness" in self.bias_type else self.pose_scenario
        
        print(f"\n--- Contingency table {stage_name}: {self.bias_type}, f={self.bias_factor}, scenario={scenario} ---")
        
        if "brightness" in self.bias_type and 'brightness' in df.columns:
            ct = pd.crosstab(df['temp_label'], [df['gender'], df['brightness']])
        elif 'pose' in df.columns:
            ct = pd.crosstab(df['temp_label'], [df['gender'], df['pose']])
        else:
            ct = pd.crosstab(df['temp_label'], df['gender'])
            
        print(ct)

        t_class = self.target_class if self.target_class is not None else "all"
        log_dir = f"dataset_logs_fase2-c{t_class}"
        os.makedirs(log_dir, exist_ok=True)
        
        filename = f"{log_dir}/{stage_name}_dist_{self.bias_type}_f{self.bias_factor}_{scenario}.csv"
        ct.to_csv(filename)

    # Funciones del DataModule
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
