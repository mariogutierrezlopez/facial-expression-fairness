#   resnet50.py
#   Mario Gutiérrez López

from typing import Any, Callable
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import optim, nn, utils, Tensor
import lightning as L
from torchvision import models
import torchmetrics
from torchmetrics import ConfusionMatrix
import torch
from torch.optim.lr_scheduler import OneCycleLR

from src.utils.losses import FocalLoss
        
class ResNet50(L.LightningModule):

    def __init__(self, n_outputs, lr=1e-3, weights_tensor=None):
        super().__init__()

        # Hyperparameters
        self.n_outputs = n_outputs
        self.lr = lr

        # Init the model
        self.model = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V1)

        #Modify last layers
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, n_outputs)

        # Loss function and metrics

        # Funcion FocalLoss autoimplementada, alpha da más importancia a unas variables que a otras para contrarrestar el desbalance
        self.loss = FocalLoss(alpha=weights_tensor, gamma=2.0)
        # self.loss = nn.CrossEntropyLoss()
        
        
        
        self.training_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        self.valid_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        
        self.save_hyperparameters()
        
        #METRICS
        #Conf matrix
        self.conf_matrix = ConfusionMatrix(num_classes=self.n_outputs, task="multiclass")
        #Recall metrics
        self.val_recall_per_class = torchmetrics.Recall(num_classes=self.n_outputs, task="multiclass", average=None)
        self.val_avg_recall = torchmetrics.Recall(num_classes=self.n_outputs, task="multiclass", average="macro")
        self.test_recall_per_class = torchmetrics.Recall(num_classes=self.n_outputs, task="multiclass", average=None)
        self.test_avg_recall = torchmetrics.Recall(num_classes=self.n_outputs, task="multiclass", average="macro")
        
        # Demographic evaluation
        self.test_step_outputs = []


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.training_acc(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        l = self.loss(logits, y)

        self.valid_acc(logits, y)
        self.val_avg_recall(logits, y)

        self.log("val_recall_macro", self.val_avg_recall, on_epoch=True, prog_bar=True)
        self.log("val_loss", l, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        gender_male = batch["meta"]["gender_male"]
        gender_female = batch["meta"]["gender_female"]

        logits = self(x)
        loss = self.loss(logits, y)

        self.test_acc(logits, y)
        self.test_recall_per_class(logits, y)
        self.test_avg_recall(logits, y)
        self.conf_matrix(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append({
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu(),
            "gender_male": gender_male.detach().cpu(),
            "gender_female": gender_female.detach().cpu()
        })

        # LOG loss and acc
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        
        self.log("test_acc_epoch", self.test_acc.compute())


        #OUTPUT CONF MATRIX
        print("Matrix de confusión\n")
        print(self.conf_matrix.compute())

        #OUTPUT RECALL METRICS
        per_class = self.test_recall_per_class.compute()
        avg_recall = self.test_avg_recall.compute()

        self.log("test_avg_recall", avg_recall)
        for i, recall_val in enumerate(per_class):
            self.log(f"test_recall_class_{i}", recall_val)
            
        if len(self.test_step_outputs) > 0:
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
            all_male = torch.cat([x["gender_male"] for x in self.test_step_outputs])
            all_female = torch.cat([x["gender_female"] for x in self.test_step_outputs])

            for c in range(self.n_outputs):
                # 1. Calcular para Hombres (gender_male == 1)
                mask_male = (all_targets == c) & (all_male == 1)
                if mask_male.any():
                    subset_preds = all_preds[mask_male]
                    subset_targets = all_targets[mask_male]
                    recall = (subset_preds == subset_targets).sum().item() / len(subset_targets)
                    self.log(f"test_recall_class_{c}_MALE", recall)

                # 2. Calcular para Mujeres (gender_female == 1)
                mask_female = (all_targets == c) & (all_female == 1)
                if mask_female.any():
                    subset_preds = all_preds[mask_female]
                    subset_targets = all_targets[mask_female]
                    recall = (subset_preds == subset_targets).sum().item() / len(subset_targets)
                    self.log(f"test_recall_class_{c}_FEMALE", recall)

            self.test_step_outputs.clear()
        
        #RESET METRICS
        self.conf_matrix.reset()
        self.test_avg_recall.reset()
        self.test_recall_per_class.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
