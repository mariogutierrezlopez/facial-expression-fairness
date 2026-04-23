# emotieff.py
# Mario Gutiérrez López

from typing import Any
from lightning import LightningModule
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import torch
import timm

from ..utils.robust_optimization import RobustOptimizer

# Metricas
import torchmetrics
from torchmetrics import ConfusionMatrix
import wandb


class EmotiEff(LightningModule):
    def __init__(self, n_outputs: int, lr: float, class_weights, model_name='tf_efficientnet_b0_ns', vggface2_weights_path=None, freeze_backbone=False, dataset_name="MultiPIE") -> None:
        super().__init__()

        self.n_outputs = n_outputs
        self.lr = lr
        self.class_weights = class_weights

        self.dataset_name = dataset_name # Nombre para guardar embeddings

        self.save_hyperparameters(ignore=['class_weights'])

        # Modelo
        self.model = timm.create_model(model_name, pretrained=False)
        
        # Cargar pesos VGGFace2
        if vggface2_weights_path:
            pesos = torch.load(vggface2_weights_path, map_location='cpu', weights_only=False)

            if not isinstance(pesos, dict):
                pesos = pesos.state_dict()

            self.model.load_state_dict(pesos, strict=False)
            
        # Adaptar el clasificador a emociones
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, self.n_outputs)

        # Congelar/Descongelar backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True
        
        self.test_acc = torchmetrics.Accuracy(num_classes=self.n_outputs, task="multiclass")
        self.conf_matrix = ConfusionMatrix(num_classes=self.n_outputs, task="multiclass")
        self.test_recall_per_class = torchmetrics.Recall(num_classes=self.n_outputs, task="multiclass", average=None)
        self.test_avg_recall = torchmetrics.Recall(num_classes=self.n_outputs, task="multiclass", average="macro")
        
        self.test_step_outputs = []
    
    def forward(self, x, return_embeddings=False):
        if return_embeddings:
            features = self.model.forward_features(x)
            embeddings = self.model.global_pool(features)
            logits = self.model.classifier(embeddings)
            return logits, embeddings
        else:
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = self.loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]
        logits = self(x)
        loss = self.loss(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, self.model.parameters()), optim.Adam, lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            cycle_momentum=False # <- Fix de compatibilidad con SAM
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

    def loss(self, pred, target):
        weights = torch.FloatTensor(list(self.class_weights.values())).to(pred.device)
        
        def label_smooth(target, n_classes: int, label_smoothing=0.1):
            batch_size = target.size(0)
            target = torch.unsqueeze(target, 1)
            soft_target = torch.zeros((batch_size, n_classes), device=target.device)
            soft_target.scatter_(1, target, 1)
            soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
            return soft_target

        def cross_entropy_loss_with_soft_target(pred, soft_target):
            return torch.mean(torch.sum(- weights * soft_target * F.log_softmax(pred, dim=-1), 1))

        soft_target = label_smooth(target, pred.size(1))
        return cross_entropy_loss_with_soft_target(pred, soft_target)

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["target"]

        meta = batch.get("meta", {})
        gender_male = meta.get("gender_male", torch.ones_like(y) * -1)
        gender_female = meta.get("gender_female", torch.ones_like(y) * -1)
        illumination = meta.get("illumination", (torch.ones_like(y).float() * -1))

        # Llamamos al forward pidiendo los embeddings
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
            "illumination": illumination.detach().cpu()
        })

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test_acc_epoch", self.test_acc.compute())

        print("\nMatriz de confusión (SOTA):")
        print(self.conf_matrix.compute())

        per_class = self.test_recall_per_class.compute()
        avg_recall = self.test_avg_recall.compute()

        self.log("test_avg_recall", avg_recall)
        for i, recall_val in enumerate(per_class):
            self.log(f"test_recall_class_{i}", recall_val)
            
        if len(self.test_step_outputs) > 0:
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
            all_embeddings = torch.cat([x["embeddings"] for x in self.test_step_outputs])
            all_illumination = torch.cat([x["illumination"] for x in self.test_step_outputs])
            all_male = torch.cat([x["gender_male"] for x in self.test_step_outputs])
            all_female = torch.cat([x["gender_female"] for x in self.test_step_outputs])

            if self.dataset_name == "MultiPIE" and wandb.run is not None:
                class_names = ["Neutral", "Smile", "Surprise", "Squint", "Disgust", "Scream"]
                wandb.log({
                    "test_confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None, y_true=all_targets.tolist(), preds=all_preds.tolist(), class_names=class_names
                    )
                })

            # Cálculo de Recall demográfico
            if (all_male != -1).any():
                for c in range(self.n_outputs):
                    # Hombres
                    mask_male = (all_targets == c) & (all_male == 1)
                    if mask_male.any():
                        subset_preds = all_preds[mask_male]
                        subset_targets = all_targets[mask_male]
                        recall = (subset_preds == subset_targets).sum().item() / len(subset_targets)
                        self.log(f"test_recall_class_{c}_MALE", recall)

                    # Mujeres
                    mask_female = (all_targets == c) & (all_female == 1)
                    if mask_female.any():
                        subset_preds = all_preds[mask_female]
                        subset_targets = all_targets[mask_female]
                        recall = (subset_preds == subset_targets).sum().item() / len(subset_targets)
                        self.log(f"test_recall_class_{c}_FEMALE", recall)
            
            # Guardar resultados
            gender_combined = torch.zeros_like(all_targets)
            gender_combined[all_male == 1] = 0
            gender_combined[all_female == 1] = 1

            torch.save({
                'embeddings': all_embeddings,
                'targets': all_targets,
                'preds': all_preds,
                'illumination': all_illumination,
                'gender': gender_combined.cpu()
            }, f'SOTA_test_embeddings_{self.dataset_name}.pt')
            print(f"Embeddings guardados exitosamente en SOTA_test_embeddings_{self.dataset_name}.pt")

            self.test_step_outputs.clear()
        
        # Resetear métricas
        self.conf_matrix.reset()
        self.test_avg_recall.reset()
        self.test_recall_per_class.reset()