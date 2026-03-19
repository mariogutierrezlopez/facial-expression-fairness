#   test.py
#   Mario Gutiérrez López


from src.data.datamodules.datamodule import MultiPIEDataModule
from src.models.resnet50 import ResNet50
from lightning.pytorch.loggers import WandbLogger
import lightning as L
import wandb


TARGET_CLASS = 2
EXPERIMENTS = [
    {"bias_type": "representational", "f": 0.0,  "ckpt": "checkpoints/Repres_bias_f0.0/best-model-epoch=08-val_loss=0.78.ckpt"},
    {"bias_type": "representational", "f": 0.25, "ckpt": "checkpoints/Repres_bias_f0.25/best-model-epoch=07-val_loss=0.52.ckpt"},
    {"bias_type": "representational", "f": 0.5,  "ckpt": "checkpoints/Repres_bias_f0.5/best-model-epoch=07-val_loss=0.55.ckpt"},
    {"bias_type": "representational", "f": 0.75, "ckpt": "checkpoints/Repres_bias_f0.75/best-model-epoch=05-val_loss=0.56.ckpt"},
    {"bias_type": "representational", "f": 1.0,  "ckpt": "checkpoints/Repres_bias_f1.0/best-model-epoch=02-val_loss=0.83.ckpt"},
    
    {"bias_type": "stereotipical", "f": 0.0,  "ckpt": "checkpoints/Stereotipical_bias_f0.0/best-model-epoch=04-val_loss=0.56.ckpt"},
    {"bias_type": "stereotipical", "f": 0.25, "ckpt": "checkpoints/Stereotipical_bias_f0.25/best-model-epoch=03-val_loss=0.58.ckpt"},
    {"bias_type": "stereotipical", "f": 0.5,  "ckpt": "checkpoints/Stereotipical_bias_f0.5/best-model-epoch=05-val_loss=0.47.ckpt"},
    {"bias_type": "stereotipical", "f": 0.75, "ckpt": "checkpoints/Stereotipical_bias_f0.75/best-model-epoch=06-val_loss=0.47.ckpt"},
    {"bias_type": "stereotipical", "f": 1.0,  "ckpt": "checkpoints/Stereotipical_bias_f1.0/best-model-epoch=06-val_loss=0.68.ckpt"}
]

for exp in EXPERIMENTS:
    b_type = exp["bias_type"]
    f_val = exp["f"]
    ckpt_path = exp["ckpt"]

    prefix = "Repres" if b_type == "representational" else "Stereotipical"
    run_name = f"{prefix}_bias_f{f_val}"

    data = MultiPIEDataModule(
        data_dir="/home12TB1/database/recognition/faces/MultiPie/data/",
        csv_path="/home12TB1/database/recognition/faces/MultiPie/demographic_info_cropped.csv",
        batch_size=256,
        num_workers=12,
        bias_type=b_type,
        bias_factor=f_val,
        target_class=TARGET_CLASS
    )

    model = ResNet50(n_outputs=6)

    logger = WandbLogger(project = "MultiPIE-Bias-Analysis", name=run_name)

    trainer = L.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        logger=logger,
        devices=1
    )

    trainer.test(model=model, datamodule=data, ckpt_path=ckpt_path)

    wandb.finish()
    print(f"Experiment {exp} finished, going for next one")
