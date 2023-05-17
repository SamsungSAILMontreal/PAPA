import torch
import torch.nn.functional as F
from utils.mixup import Mixup

# From https://github.com/rwightman/pytorch-image-models

class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss

# One-hot the target, use mixup (optional), use smoothing (optional)
class TransformTarget(torch.nn.Module):
    def __init__(self, num_classes):
        super(TransformTarget, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, y, mixup=0.0, smoothing=0.0, cutmix=0.0):
        my_mixup = Mixup(mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=None, prob=1.0, switch_prob=0.5, 
            mode='batch', correct_lam=True, label_smoothing=smoothing, num_classes=self.num_classes)
        x, y = my_mixup(x, y)
        return x, y
