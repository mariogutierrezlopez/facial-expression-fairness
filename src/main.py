#   main.py
#   Mario Gutiérrez López

import argparse
import lightning as L
import torch
from src.models.resnet50 import ResNet50

from src.data.datamodules.multipie_dm import MultiPIEDataModule
from src.data.datamodules.affectnet_dm import AffectNetDataModule
from src.data.datamodules.affwild2_dm import AffWild2DatModule

from src.utils.utils import calc_nlimits

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

# --- CONFIGURACIÓN MULTIPIE ---
MPIE_DATA_DIR = "/home12TB1/database/recognition/faces/MultiPie/data/"
MPIE_CSV_PATH = "/home12TB1/database/recognition/faces/MultiPie/demographic_info_cropped.csv"

# --- CONFIGURACIÓN AFFECTNET ---
AFFNET_DATA_DIR = "/home12TB1/database/recognition/faces/affectnet/"
AFFNET_CSV_TRAIN_PATH = "/home12TB1/database/recognition/faces/affectnet/affectnetplus_train_annotations.csv"
AFFNET_CSV_TEST_PATH = "/home12TB1/database/recognition/faces/affectnet/affectnetplus_test_annotations_quality_illum.csv"

# --- CONFIGURACIÓN AFFWILD2 ---
AFFWILD2_DATA_DIR = "/home12TB1/database/recognition/faces/affwild2/"
AFFWILD2_CSV_TRAIN_PATH = "/home12TB1/database/recognition/faces/affwild2/dataframe_train.csv"
AFFWILD2_CSV_TEST_PATH = "/home12TB1/database/recognition/faces/affwild2/dataframe_val_pose_demographic_illum.csv"

# --- HIPERPARÁMETROS GLOBALES ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_WORKERS = 12
MAX_EPOCHS = 50
EPOCHS_MPIE = 20
N_OUTPUTS_MPIE = 6
N_OUTPUTS_AFFNET = 8
N_OUTPUTS_AFFWILD2 = 8
BIAS_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0]

# class weights focal loss
num_samples = [17114, 45567, 5037, 3618, 1454, 599, 2303, 908]
total = sum(num_samples)
n_classes = len(num_samples)
weights = [total / (n_classes * n) for n in num_samples]
class_weights = torch.tensor(weights, dtype=torch.float32)


def run_multipie_experiment(exp_name, bias_type, bias_factor, n_limit, target_class=None):

    if wandb.run is not None:
        wandb.finish()

    data = MultiPIEDataModule(
        data_dir=MPIE_DATA_DIR,
        csv_path=MPIE_CSV_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        bias_type=bias_type,
        bias_factor=bias_factor,
        n_limit=n_limit,
        target_class=target_class
    )

    model = ResNet50(n_outputs=N_OUTPUTS_MPIE, lr=LEARNING_RATE)

    logger = WandbLogger(
        name=exp_name,
        project="MultiPIE_Stereotypical_All_Classes",
        log_model="all",
    )

    print(f"Hiperparámetros del modelo{model.hparams}")
    
    checkpoint_dir = f"checkpoints/{exp_name}/"
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=0, dirpath=f"checkpoints/{exp_name}/")

    trainer = L.Trainer(
        accelerator="gpu",
        # devices=[0],
        precision="16-mixed",
        max_epochs=EPOCHS_MPIE,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(model=model, datamodule=data)

    trainer.test(model=model, datamodule=data, ckpt_path="last")

    wandb.finish()

# AFFECTNET
def run_affectnet_baseline():
    exp_name = "AffectNet_fl_balanced"
    if wandb.run is not None: wandb.finish()

    data = AffectNetDataModule(
        data_dir=AFFNET_DATA_DIR, 
        csv_train_path=AFFNET_CSV_TRAIN_PATH,
        csv_test_path=AFFNET_CSV_TEST_PATH, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    model = ResNet50(n_outputs=N_OUTPUTS_AFFNET, lr=LEARNING_RATE, weights_tensor=class_weights)
    logger = WandbLogger(name=exp_name, project="AffectNet_Baseline", log_model="all")
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall_macro',
        dirpath='checkpoints_affectnet_fl_balanced/',
        filename='restnet50-epoch{epoch:02d}-val{val_recall_macro:.3f}',
        save_top_k=2,
        mode='max',
        verbose=True
    )

    trainer = L.Trainer(accelerator="gpu", precision="16-mixed", max_epochs=MAX_EPOCHS,
                        log_every_n_steps=10, logger=logger, callbacks=[checkpoint_callback])
    
    print("Iniciando entrenamiento Baseline de AffectNet")
    trainer.fit(model=model, datamodule=data)
    trainer.test(model=model, datamodule=data, ckpt_path="best")
    wandb.finish()

# AFFWILD2
def run_affwild2_baseline():
    exp_name = "AffWild2_baseline"
    if wandb.run is not None: wandb.finish()

    data = AffWild2DatModule(
        data_dir=AFFWILD2_DATA_DIR, 
        csv_train_path=AFFWILD2_CSV_TRAIN_PATH,
        csv_val_path=AFFWILD2_CSV_TEST_PATH, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )


    model = ResNet50(n_outputs=N_OUTPUTS_AFFWILD2, lr=LEARNING_RATE, weights_tensor=None)
    logger = WandbLogger(name=exp_name, project="AffWild2", log_model="all")
    

    checkpoint_callback = ModelCheckpoint(
        monitor='val_recall_macro',
        dirpath='checkpoints_affwild2_baseline/',
        filename='restnet50-epoch{epoch:02d}-val{val_recall_macro:.3f}',
        save_top_k=2,
        mode='max',
        verbose=True
    )

    trainer = L.Trainer(accelerator="gpu", precision="16-mixed", max_epochs=MAX_EPOCHS,
                        log_every_n_steps=10, logger=logger, callbacks=[checkpoint_callback])
    
    print("Iniciando entrenamiento Baseline de Affwild2")
    trainer.fit(model=model, datamodule=data)
    trainer.test(model=model, datamodule=data, ckpt_path="best")
    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="TFM Face Expression Bias")
    parser.add_argument("--dataset", type=str, choices=["multipie", "affectnet", "affwild2"], default="affectnet",
                        help="Elige qué dataset entrenar (multipie o affectnet)")
    args = parser.parse_args()

    if args.dataset == "multipie":
        print("LANZANDO EXPERIMENTOS MULTIPIE")
        n_limit_repres, n_limit_stereo = calc_nlimits(MPIE_CSV_PATH, BIAS_FACTORS)
        
        TARGET_CLASS_IDS = [0,1,2,3,4,5]
        for target_id in TARGET_CLASS_IDS:
            for f in BIAS_FACTORS:
                run_multipie_experiment(
                    exp_name=f"Stereotipical_bias_c{target_id}_f{f}",
                    bias_type="stereotipical",
                    bias_factor=f,
                    target_class=target_id,
                    n_limit=n_limit_stereo
                )
    
    elif args.dataset == "affectnet":
        print("LANZANDO BASELINE AFFECTNET")
        run_affectnet_baseline()
    
    elif args.dataset == "affwild2":
        print("LANZANDO EXPERIMENTO AFFWILD2")
        run_affwild2_baseline()