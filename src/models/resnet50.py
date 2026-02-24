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

class ResNet50(L.LightningModule):

    def __init__(self, n_outputs, lr=1e-3):
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

        self.loss = nn.CrossEntropyLoss()
        self.training_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        self.valid_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        self.test_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        
        self.save_hyperparameters()
        
        #METRICS
        #Conf matrix
        self.conf_matrix = ConfusionMatrix(num_classes=self.n_outputs, task="multiclass")
        #Recall metrics
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
        acc = self.valid_acc(logits, y)
        self.log("val_loss", l, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        genders = batch["meta"]["gender"]

        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.test_acc(logits,y)

        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append({
            "preds": preds.detach.cpu(),
            "targets": y.detach().cpu(),
            "genders": genders
        })

        # LOG loss and acc
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # COMPUTE METRICS
        self.test_recall_per_class.update(logits, y)
        self.test_avg_recall.update(logits, y)
        self.conf_matrix.update(logits, y)

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

        # DEMOGRAPHIC RECALL CALCULATION
        if len(self.test_step_outputs) > 0:
            # Flatten lists
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
            all_genders = [g for x in self.test_step_outputs for g in x["genders"]]

            for c in range(self.n_outputs):
                for gender_label in ["Male", "Female"]:
                    # Create boolean mask for this specific class and gender
                    mask = torch.tensor([
                        (all_targets[i].item() == c) and (all_genders[i] == gender_label) 
                        for i in range(len(all_targets))
                    ])

                    # Evitar division por cero
                    if not mask.any():
                        continue

                    # Filter using the mask
                    subset_preds = all_preds[mask]
                    subset_targets = all_targets[mask]

                    # Recall = True Positives / Total Actual Positives
                    correct = (subset_preds == subset_targets).sum().item()
                    total = len(subset_targets)
                    recall = correct / total

                    # Log to WandB!
                    self.log(f"test_recall_class_{c}_{gender_label.upper()}", recall)

            self.test_step_outputs.clear()
        
        #RESET METRICS
        self.conf_matrix.reset()
        self.test_avg_recall.reset()
        self.test_recall_per_class.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
