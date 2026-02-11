#   test.py
#   Mario Gutiérrez López


from src.data.datamodule import MultiPIEDataModule
from src.models.resnet50 import ResNet50
from lightning.pytorch.loggers import WandbLogger
import lightning as L

data = MultiPIEDataModule(
    data_dir="/home12TB1/database/recognition/faces/MultiPie/data/",
    csv_path="/home12TB1/database/recognition/faces/MultiPie/demographic_info_cropped.csv",
    batch_size=256,
    num_workers=12
)

model = ResNet50(n_outputs=6)

logger = WandbLogger(project = "MultiPIE_test_results", name="MultiPIE_run")

trainer = L.Trainer(
    accelerator="gpu",
    precision="16-mixed",
    logger=logger,
    devices=1
)

CKPT_PATH = "checkpoints/best-model-epoch=08-val_loss=0.34.ckpt"
trainer.test(model=model, datamodule=data, ckpt_path=CKPT_PATH)