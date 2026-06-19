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
import wandb
import cv2
import numpy as np

#Gradcam
from pytorch_grad_cam import GuidedBackpropReLUModel, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.utils.losses import FocalLoss
        
class ResNet50(L.LightningModule):

    def __init__(self, n_outputs, lr=1e-3, weights_tensor=None, dataset_name:str = None):
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

        # # Funcion FocalLoss autoimplementada, alpha da más importancia a unas variables que a otras para contrarrestar el desbalance
        if weights_tensor is not None:
            self.register_buffer('class_weights', weights_tensor)
            self.loss = FocalLoss(alpha=self.class_weights, gamma=2.0)
        else:
            self.loss = FocalLoss(alpha=None, gamma=2.0)
        
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

        # Dataset name for saving test results
        self.dataset_name = dataset_name

    # La función fordward ha sido modificada para obtener los embeddings de la imagen para el Hito 3
    # Cuando el flag 'return_embeddings' == True
    def forward(self, x, return_embeddings=False):
        # Pasar la imagen por todas las capas de la ResNet excepto la última
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        embeddings = torch.flatten(x, 1)

        # Pasar el embedding por la capa final para obtener la predicción
        logits = self.model.fc(embeddings)

        if return_embeddings:
            return logits, embeddings
        else:
            return logits
    
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

        meta = batch.get("meta", {})

        gender_male = meta.get("gender_male", torch.ones_like(y) * -1)
        gender_female = meta.get("gender_female", torch.ones_like(y) * -1)
        illumination = meta.get("illumination", (torch.ones_like(y).float() * -1))
        pose = meta.get("pose", torch.zeros((x.size(0), 3)))


        logits, embeddings = self(x, return_embeddings=True)
        loss = self.loss(logits, y)

        self.test_acc(logits, y)
        self.test_recall_per_class(logits, y)
        self.test_avg_recall(logits, y)
        self.conf_matrix(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.test_step_outputs.append({
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu(),
            "embeddings": embeddings.detach().cpu(),
            "gender_male": gender_male.detach().cpu(),
            "gender_female": gender_female.detach().cpu(),
            "illumination": illumination.detach().cpu(),
            "pose": pose.detach().cpu()
        })

        # LOG loss and acc
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        
        self.log("test_acc_epoch", self.test_acc.compute())


        #OUTPUT CONF MATRIX
        print("Matrix de confusión\n")
        final_conf_matrix = self.conf_matrix.compute()
        print(final_conf_matrix)
        
        print("Matriz guardada nativamente en WandB.")

        #OUTPUT RECALL METRICS
        per_class = self.test_recall_per_class.compute()
        avg_recall = self.test_avg_recall.compute()

        self.log("test_avg_recall", avg_recall)
        for i, recall_val in enumerate(per_class):
            self.log(f"test_recall_class_{i}", recall_val)
            
        if len(self.test_step_outputs) > 0:
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

            if self.dataset_name == "MultiPIE":
                class_names = ["Neutral", "Smile", "Surprise", "Squint", "Disgust", "Scream"]
                wandb.log({
                    "test_confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_targets.tolist(),
                        preds=all_preds.tolist(),
                        class_names=class_names
                    )
                })
                print("Matriz interactiva guardada en WandB.")

            all_embeddings = torch.cat([x["embeddings"] for x in self.test_step_outputs])
            all_illumination = torch.cat([x["illumination"] for x in self.test_step_outputs])

            all_pose = torch.cat([x["pose"] for x in self.test_step_outputs])

            # Recall por género
            all_male = torch.cat([x["gender_male"] for x in self.test_step_outputs])
            all_female = torch.cat([x["gender_female"] for x in self.test_step_outputs])

            # Control para affwild2 (no tiene etiquetas de género)
            if (all_male != -1).any():
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
            
            # Información de género para embeddings
            gender_combined = torch.zeros_like(all_targets)
            gender_combined[all_male == 1] = 0
            gender_combined[all_female == 1] = 1

            # Guardar los tensores para el análisis de distancias
            torch.save({
                'embeddings': all_embeddings,
                'targets': all_targets,
                'preds': all_preds,
                'illumination': all_illumination,
                'gender': gender_combined.cpu(),
                "pose": all_pose
            }, f'test_embeddings_results_{self.dataset_name}.pt')
            print(f"Embeddings guardados en test_embeddings_results_{self.dataset_name}.pt")


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

    # GRAD-CAM

    def get_gradcam(self, x, target_category=None):
        target_layers = [self.model.layer3[-1], self.model.layer4[-1]]
            
        cam = LayerCAM(model=self, target_layers=target_layers)
        
        # Si no se especifica clase, explicar la predicción del modelo
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        
        # Generar el mapa (grayscale)
        grayscale_cam = cam(input_tensor=x, targets=targets, aug_smooth=True, eigen_smooth=True) # type: ignore
        grayscale_cam = grayscale_cam[0, :]
        
        return grayscale_cam

    def visualize_gradcam(self, x, target_category=None):
        """
        Devuelve una imagen RGB con el mapa de calor superpuesto
        """
        self.eval()
        grayscale_cam = self.get_gradcam(x, target_category)
        
        # Preparar la imagen original para visualización, hay que des-normalizarla o pasarla a [0, 1] y HWC
        img_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return visualization

    def visualize_guided_gradcam(self, x, target_category=None):
        self.eval()

        grayscale_cam = self.get_gradcam(x, target_category)
        
        gb_model = GuidedBackpropReLUModel(model=self, device=self.device)
        
        gb = gb_model(input_img=x, target_category=target_category)
        
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = gb * cam_mask
        
        cam_gb = (cam_gb - np.min(cam_gb)) / (np.max(cam_gb) - np.min(cam_gb) + 1e-7)
        
        return cam_gb