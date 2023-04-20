import torch
import torch.nn.functional as F

# From https://github.com/rwightman/pytorch-image-models

class SoftTargetCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target1, target2, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target1, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target2, num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


# One-hot the target, use mixup (optional), use smoothing (optional)
class TransformTarget(torch.nn.Module):
    def __init__(self, num_classes):
        super(TransformTarget, self).__init__()
        self.num_classes = num_classes

    def forward(self, x, y, mixup=0.0, smoothing=0.0):
        if mixup > 0.0:
            b = torch.ones(y.shape[0], device=y.device)*mixup
            lam = torch.distributions.beta.Beta(b, b).sample()
            # mix indexes
            mix_index = torch.randperm(y.shape[0], device=y.device)
            # Change x and y for mixup
            y = mixup_target(y, y[mix_index], self.num_classes, lam=lam.view(-1,1), smoothing=smoothing)
            x = lam.view(-1,1,1,1) * x + (1-lam.view(-1,1,1,1)) * x[mix_index]
        else:
            y = mixup_target(y, y, self.num_classes, lam=0.0, smoothing=smoothing)
        return x, y
