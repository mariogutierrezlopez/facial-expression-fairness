#   main.py
#   Mario Gutiérrez López

import lightning as L
import torch
from src.models.resnet50 import ResNet50
from src.data.datamodule import MultiPIEDataModule
from src.utils.utils import calc_nlimits

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

# CONFIGURACION
DATA_DIR = "/home12TB1/database/recognition/faces/MultiPie/data/"
CSV_PATH = "/home12TB1/database/recognition/faces/MultiPie/demographic_info_cropped.csv"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 12
MAX_EPOCHS = 20
N_OUTPUTS = 6
#Factores de sesgo f
BIAS_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_experiment(exp_name, bias_type, bias_factor, n_limit, target_class=None):

    if wandb.run is not None:
        wandb.finish()

    data = MultiPIEDataModule(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        bias_type=bias_type,
        bias_factor=bias_factor,
        n_limit=n_limit,
        target_class=target_class
    )

    model = ResNet50(n_outputs=N_OUTPUTS, lr=LEARNING_RATE)

    logger = WandbLogger(
        name=exp_name,
        project="MultiPIE_Stereotypical_All_Classes",
        log_model="all",
    )

    print(f"Hiperparámetros del modelo{model.hparams}")
    
    checkpoint_dir = f"checkpoints/{exp_name}/"
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        dirpath=checkpoint_dir,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        # devices=[0],
        precision="16-mixed",
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model=model, datamodule=data)

    trainer.test(model=model, datamodule=data, ckpt_path="last")

    wandb.finish()

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    n_limit_repres, n_limit_stereo = calc_nlimits(CSV_PATH, BIAS_FACTORS)
    print(f"Limite representacional: {n_limit_repres}")
    print(f"Límite estereotípico: {n_limit_stereo}")

    global_n_limit = min(n_limit_repres, n_limit_stereo)
    print(f"N_limit global para ambos: {global_n_limit}")
    
    print("Analizando sesgos estereotípicos")

    # TARGET_CLASS_ID = 3
    TARGET_CLASS_IDS = [0,1,2,3,4,5]
    for target_id in TARGET_CLASS_IDS:
        for f in BIAS_FACTORS:
            run_experiment(
                exp_name=f"Stereotipical_bias_c{target_id}_f{f}",
                bias_type="stereotipical",
                bias_factor=f,
                target_class=target_id,
                n_limit=n_limit_stereo
            )
            wandb.finish()