import lightning as L
import torch
from src.models.resnet50 import ResNet50
from src.data.datamodule import MultiPIEDataModule

from lightning.pytorch.loggers import WandbLogger


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    # LightningDataModule y LightningModule
    data = MultiPIEDataModule(
        data_dir="/home12TB1/database/recognition/faces/MultiPie/data/",
        csv_path="/home12TB1/database/recognition/faces/MultiPie/demographic_info_cropped.csv",
        batch_size=256,
        num_workers=12
    )

    model = ResNet50(n_outputs=6, lr=2e-3)

    logger = WandbLogger(log_model="all")

    print(f"Hiperparámetros del modelo{model.hparams}")

    trainer = L.Trainer(
        accelerator="gpu",
        # devices=[0],
        precision="16-mixed",
        max_epochs=10,
        log_every_n_steps=10,
        logger=logger
    )

    trainer.fit(model=model, datamodule=data)