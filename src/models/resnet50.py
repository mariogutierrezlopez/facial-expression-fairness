from typing import Any, Callable
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import optim, nn, utils, Tensor
import lightning as L
from torchvision import models
import torchmetrics
from torchmetrics import ConfusionMatrix

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

        self.conf_matrix = ConfusionMatrix(num_classes=self.n_outputs, task="multiclass")


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
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.test_acc(logits,y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.conf_matrix.update(logits, y)

    def on_test_epoch_end(self):
        print("Matrix de confusión\n")
        print(self.conf_matrix.compute())
        self.conf_matrix.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
