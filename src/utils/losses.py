import torch.nn.functional as F
import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)
        :param alpha: Pesos para cada clase (puedes pasarle un tensor)
        :param gamma: Factor de enfoque para ejemplos difíciles
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculamos la Cross Entropy básica primero
        weights = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weights)
        
        # p_t es la probabilidad de la clase correcta
        pt = torch.exp(-ce_loss) 
        
        # Aplicamos la fórmula de Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        