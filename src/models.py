from typing import Any, Callable
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import optim, nn, utils, Tensor
import lightning as L
from torchvision import models
import torchmetrics

class ResNet50(L.LightningModule):

    def __init__(self, n_outputs, lr,):
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

        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.training_acc(logits, y)
        self.log("training loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("training acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch
        logits = self(x)
        l = self.loss(logits, y)
        acc = self.valid_acc(logits, y)
        self.log("valid_loss", l, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
