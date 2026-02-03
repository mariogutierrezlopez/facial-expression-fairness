import lightning as L
import torch
from src.models import ResNet50
from src.data import FakeDataModule

from lightning.pytorch.loggers import WandbLogger


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    # LightningDataModule y LightningModule
    data = FakeDataModule(data_dir='./data', batch_size=128, num_workers=39)
    model = ResNet50(n_outputs=10, lr=1e-3)

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