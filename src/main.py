#   main.py
#   Mario Gutiérrez López

import argparse
import lightning as L
import torch
from src.models.resnet50 import ResNet50
from src.models.emotieff import EmotiEff

from src.data.datamodules.multipie_dm import MultiPIEDataModule
from src.data.datamodules.affectnet_dm import AffectNetDataModule
from src.data.datamodules.affwild2_dm import AffWild2DatModule

from src.utils.utils import calc_nlimits, calc_nlimit_pose

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import wandb

# --- CONFIGURACIÓN MULTIPIE ---
MPIE_DATA_DIR = "/home12TB1/database/recognition/faces/MultiPie/data/"
# MPIE_CSV_PATH = "/home12TB1/database/recognition/faces/MultiPie/demographic_labels.csv"
MPIE_CSV_PATH = "/home/mgutierrez/TFM/src/data/datasets/multipie_70_15_15.csv"
# --- CONFIGURACIÓN AFFECTNET ---
AFFNET_DATA_DIR = "/home12TB1/database/recognition/faces/affectnet/"
AFFNET_CSV_TRAIN_PATH = "/home12TB1/database/recognition/faces/affectnet/affectnetplus_train_annotations.csv"
AFFNET_CSV_TEST_PATH = "/home12TB1/database/recognition/faces/affectnet/affectnetplus_test_annotations_quality_illum.csv"

# --- CONFIGURACIÓN AFFWILD2 ---
AFFWILD2_DATA_DIR = "/home12TB1/database/recognition/faces/affwild2/"
AFFWILD2_CSV_TRAIN_PATH = "/home12TB1/database/recognition/faces/affwild2/dataframe_train.csv"
AFFWILD2_CSV_TEST_PATH = "/home12TB1/database/recognition/faces/affwild2/dataframe_val_pose_demographic_illum.csv"

# --- HIPERPARÁMETROS GLOBALES ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 8
MAX_EPOCHS = 100
EPOCHS_MPIE = 50
N_OUTPUTS_MPIE = 6
N_OUTPUTS_AFFNET = 8
N_OUTPUTS_AFFWILD2 = 8
BIAS_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0]

# --- CLASS WEIGHTS MULTIPIE ---
dummy_class_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}

# --- MODEL ROUTE ---
EFFICIENTNET_B0_VGAF_CHECKPOINT = "/home/mgutierrez/TFM/src/models/enet_b0_8_best_vgaf.pt"

# EXPERIMENTO ALTERACIÓN BRIGHTNESS
brightness_variants = {
    "normal": 1.0,
    "m25": 0.75,
    "m50": 0.50,
    "p25": 1.25,
    "p50": 1.50
}

# class weights focal loss
num_samples = [17114, 45567, 5037, 3618, 1454, 599, 2303, 908]
total = sum(num_samples)
n_classes = len(num_samples)
weights = [total / (n_classes * n) for n in num_samples]
class_weights = torch.tensor(weights, dtype=torch.float32)


def run_multipie_experiment(exp_name, bias_type, bias_factor=0.5, target_class=None, test_only=False, ckpt_path=None, sota_test=False, pose_scenario=""):

    if wandb.run is not None:
        wandb.finish()

    data = MultiPIEDataModule(
        data_dir=MPIE_DATA_DIR,
        csv_path=MPIE_CSV_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        bias_type=bias_type,
        bias_factor=bias_factor,
        target_class=target_class,
        pose_scenario=pose_scenario
    )

    if sota_test:
        print("Entrenando MultiPIE con EmotiEff")
        model = EmotiEff(
            n_outputs=N_OUTPUTS_MPIE,
            class_weights=dummy_class_weights,
            lr=LEARNING_RATE,
            freeze_backbone=False,
            vggface2_weights_path="/home/mgutierrez/TFM/src/models/enet_b0_8_best_vgaf.pt"
        )
    else:
        model = ResNet50(n_outputs=N_OUTPUTS_MPIE, lr=LEARNING_RATE, dataset_name="MultiPIE")

    logger = WandbLogger(
        name=exp_name,
        project="MultiPIE_emotieff_fully_balanced_vggface2",
        log_model="all",
    )

    print(f"Hiperparámetros del modelo{model.hparams}")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f"checkpoints/{exp_name}/",
        filename='emotieff-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    trainer = L.Trainer(
        accelerator="gpu",
        # devices=[0],
        precision="32-true",
        max_epochs=EPOCHS_MPIE,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    if not test_only:
        trainer.fit(model=model, datamodule=data)
        trainer.test(model=model, datamodule=data, ckpt_path="last")
    else:
        print(f"Iniciando modo test sobre MultiPIE con el modelo {ckpt_path}")
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)

    wandb.finish()

# AFFECTNET
def run_affectnet_baseline(test_only=False, ckpt_path=None, brightness=False):
    exp_name = "AffectNet_fl_balanced"
    if wandb.run is not None: wandb.finish()

    data = AffectNetDataModule(
        data_dir=AFFNET_DATA_DIR, 
        csv_train_path=AFFNET_CSV_TRAIN_PATH,
        csv_test_path=AFFNET_CSV_TEST_PATH, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    
    model = ResNet50(n_outputs=N_OUTPUTS_AFFNET, lr=LEARNING_RATE, weights_tensor=class_weights, dataset_name="Affectnet")
    logger = WandbLogger(name=exp_name, project="AffectNet_Baseline", log_model="all")
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall_macro',
        dirpath='checkpoints_affectnet_fl_balanced/',
        filename='restnet50-epoch{epoch:02d}-val{val_recall_macro:.3f}',
        save_top_k=2,
        mode='max',
        verbose=True
    )

    trainer = L.Trainer(accelerator="gpu", precision="32-true", max_epochs=MAX_EPOCHS,
                        log_every_n_steps=10, logger=logger, callbacks=[checkpoint_callback])
    
    if not test_only:
        print("Iniciando entrenamiento Baseline de AffectNet")
        trainer.fit(model=model, datamodule=data)
        trainer.test(model=model, datamodule=data, ckpt_path="best")
    elif not brightness:
        print(f"Iniciando test en AffectNet sobre el modelo {ckpt_path}")
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
    else:
        print("Iniciando experimentos en AffectNet con variaciones de iluminación")
        for variant_name, factor in brightness_variants.items():
            data = AffectNetDataModule(
                data_dir=AFFNET_DATA_DIR, 
                csv_train_path=AFFNET_CSV_TRAIN_PATH,
                csv_test_path=AFFNET_CSV_TEST_PATH, 
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                test_brightness_factor=factor
            )
            model.dataset_name = f"AffectNet_{variant_name}"

            trainer.test(model, datamodule=data, ckpt_path=ckpt_path)
    wandb.finish()

# AFFWILD2
def run_affwild2_baseline(test_only=False, ckpt_path=None, brightness=False):
    exp_name = "AffWild2_baseline"
    if wandb.run is not None: wandb.finish()

    data = AffWild2DatModule(
        data_dir=AFFWILD2_DATA_DIR, 
        csv_train_path=AFFWILD2_CSV_TRAIN_PATH,
        csv_val_path=AFFWILD2_CSV_TEST_PATH, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )


    model = ResNet50(n_outputs=N_OUTPUTS_AFFWILD2, lr=LEARNING_RATE, weights_tensor=None, dataset_name='AffWild2')
    logger = WandbLogger(name=exp_name, project="AffWild2", log_model="all")
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall_macro',
        dirpath='checkpoints_affwild2_baseline/',
        filename='restnet50-epoch{epoch:02d}-val{val_recall_macro:.3f}',
        save_top_k=2,
        mode='max',
        verbose=True
    )

    trainer = L.Trainer(accelerator="gpu", precision="32-true", max_epochs=MAX_EPOCHS,
                        log_every_n_steps=10, logger=logger, callbacks=[checkpoint_callback])
    
    if not test_only:
        print("Iniciando entrenamiento Baseline de Affwild2")
        trainer.fit(model=model, datamodule=data)
        trainer.test(model=model, datamodule=data, ckpt_path="best")
    elif not brightness:
        print(f"Iniciando test en AffWild2 sobre el modelo {ckpt_path}")
        trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)
    else:
        print("Iniciando experimentos en AffectNet con variaciones de iluminación")
        for variant_name, factor in brightness_variants.items():
            data = AffWild2DatModule(
                data_dir=AFFWILD2_DATA_DIR, 
                csv_train_path=AFFWILD2_CSV_TRAIN_PATH,
                csv_val_path=AFFWILD2_CSV_TEST_PATH, 
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                test_brightness_factor=factor
            )
            model.dataset_name = f"AffWild2_{variant_name}"

            trainer.test(model, datamodule=data, ckpt_path=ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="TFM Face Expression Bias")
    
    # Argumento para escoger dataset
    parser.add_argument("--dataset", type=str, choices=["multipie", "affectnet", "affwild2"], default="affectnet",
                        help="Elige qué dataset entrenar (multipie, affectnet o affwild2)")
    
    # Argumentos para hacer solo testing y guardar los embeddings
    parser.add_argument("--test_only", action="store_true",
                        help="Activa este flag para realizar solo el test de un modelo guardado")
    
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Ruta al archivo .ckpt del modelo entrenado")
    
    parser.add_argument("--brightness", action="store_true",
                        help="Realiza experimentos con variantes de iluminación en AffectNet")
    
    parser.add_argument("--sota", action="store_true",
                        help="Activa el Smoke Test del nuevo modelo EmotiEff en MultiPIE")
    
    parser.add_argument("--num_splits", type=int, default=1,
                        help="En cuántas terminales/partes totales vas a dividir el trabajo (ej. 3)")
    
    parser.add_argument("--split_idx", type=int, default=0,
                        help="El índice de esta terminal (0, 1, 2...)")
    
    args = parser.parse_args()

    if args.test_only and args.ckpt_path is None:
        parser.error("Si usas --test_only necesitas especificar la ruta del checkpoint")

    if args.dataset == "multipie":

        print("LANZANDO EXPERIMENTOS MULTIPIE")
        
        # 1. RECOPILAR TODOS LOS EXPERIMENTOS POSIBLES
        all_experiments = []
        

                
        # Pose (2)
        POSIBLE_POSE_VALUES = ["H_Frontal_M_Profile","M_Frontal_H_Profile"]
        for pose in POSIBLE_POSE_VALUES:
            all_experiments.append({
                "exp_name": f"Pose_{pose}",
                "bias_type": "pose",
                "bias_factor": 0.5, # Valor por defecto que tuvieras
                "target_class": None,
                "pose_scenario": pose
            })
        
        # Representacionales (5)
        for f in BIAS_FACTORS:
            all_experiments.append({
                "exp_name": f"Representational_bias_f{f}",
                "bias_type": "representational",
                "bias_factor": f,
                "target_class": None,
                "pose_scenario": ""
            })
            
        # # Estereotípicos (30)
        TARGET_CLASS_IDS = [0,1,2,3,4,5]
        for target_id in TARGET_CLASS_IDS:
            for f in BIAS_FACTORS:
                all_experiments.append({
                    "exp_name": f"Stereotipical_bias_c{target_id}_f{f}",
                    "bias_type": "stereotipical",
                    "bias_factor": f,
                    "target_class": target_id,
                    "pose_scenario": ""
                })

        # 2. FILTRAR SOLO LOS QUE LE TOCAN A ESTA TERMINAL
        my_experiments = all_experiments[args.split_idx :: args.num_splits]
        
        print(f"Total de experimentos a ejecutar en esta terminal: {len(my_experiments)} / {len(all_experiments)}")
        
        # 3. EJECUTAR EL BUCLE
        for exp in my_experiments:
        # for exp in all_experiments:
            print("="*80)
            print(f"INICIANDO EXPERIMENTO: {exp['exp_name']}")
            print("="*80)
            
            current_ckpt = f"checkpoints/{exp['exp_name']}/last.ckpt" if args.test_only else args.ckpt_path
            
            run_multipie_experiment(
                exp_name=exp['exp_name'],
                bias_type=exp['bias_type'],
                bias_factor=exp['bias_factor'],
                target_class=exp['target_class'],
                test_only=args.test_only,
                ckpt_path=current_ckpt,
                sota_test=args.sota,
                pose_scenario=exp['pose_scenario'],
            )


    elif args.dataset == "affectnet":
        print("LANZANDO BASELINE AFFECTNET")
        run_affectnet_baseline(test_only=args.test_only, ckpt_path=args.ckpt_path, brightness=args.brightness)
    
    elif args.dataset == "affwild2":
        print("LANZANDO EXPERIMENTO AFFWILD2")
        run_affwild2_baseline(test_only=args.test_only, ckpt_path=args.ckpt_path, brightness=args.brightness)