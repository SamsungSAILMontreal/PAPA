## Code for permutation alignment using feature-matching as done in the Git Re-Basin paper
# Code taken mostly from https://github.com/KellerJordan/REPAIR
# As mentionned in the paper, the code is only for VGG and Resnet18 models, generalization is highly non-trivial

import torch
from torch import nn
import torch.nn.functional as F
import scipy.optimize
import numpy as np

# Given two networks net0, net1 which each output a feature map of shape NxCxWxH,
# this will reshape both outputs to (N*W*H)xC
# and then compute a CxC correlation matrix between the two
def run_corr_matrix(device, train_dset, net0, net1, batch_size=500, n_iter=99999, shuffle=True):
    train_aug_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    n = len(train_aug_loader)
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for i, (images, _) in enumerate(train_aug_loader):
            
            img_t = images.float().to(device)
            out0 = net0(img_t).double()
            out0 = out0.permute(0, 2, 3, 1).reshape(-1, out0.shape[1])
            out1 = net1(img_t).double()
            out1 = out1.permute(0, 2, 3, 1).reshape(-1, out1.shape[1])

            # save batchwise first+second moments and outer product
            mean0_b = out0.mean(dim=0)
            mean1_b = out1.mean(dim=0)
            sqmean0_b = out0.square().mean(dim=0)
            sqmean1_b = out1.square().mean(dim=0)
            outer_b = (out0.T @ out1) / out0.shape[0]
            if i == 0:
                mean0 = torch.zeros_like(mean0_b)
                mean1 = torch.zeros_like(mean1_b)
                sqmean0 = torch.zeros_like(sqmean0_b)
                sqmean1 = torch.zeros_like(sqmean1_b)
                outer = torch.zeros_like(outer_b)
            mean0 += mean0_b / n
            mean1 += mean1_b / n
            sqmean0 += sqmean0_b / n
            sqmean1 += sqmean1_b / n
            outer += outer_b / n
            if i >= n_iter:
                break
        net0.train()
        net1.train()
    cov = outer - torch.outer(mean0, mean1)
    std0 = (sqmean0 - mean0**2).sqrt()
    std1 = (sqmean1 - mean1**2).sqrt()
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr

def get_layer_perm1(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    try:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
        assert (row_ind == np.arange(len(corr_mtx_a))).all()
        perm_map = torch.tensor(col_ind).long()
    except: # In case of problem, we do not permute
        perm_map = torch.arange(start=0, end=len(corr_mtx_a)).long()

    return perm_map

# returns the channel-permutation to make layer1's activations most closely
# match layer0's.
def get_layer_perm(device, train_dset, net0, net1, batch_size=500, n_iter=99999, shuffle=True):
    corr_mtx = run_corr_matrix(device, train_dset, net0, net1, batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    return get_layer_perm1(corr_mtx)

# modifies the weight matrices of a convolution and batchnorm
# layer given a permutation of the output channels
def permute_output(perm_map, conv, bn=None, group=False):
    pre_weights = [
        conv.weight,
    ]
    if conv.bias is not None:
        pre_weights.append(conv.bias)
    if bn is not None:
        if isinstance(bn, nn.GroupNorm):
            pre_weights.extend([
                bn.weight,
                bn.bias,
            ])
        else:
            pre_weights.extend([
                bn.weight,
                bn.bias,
                bn.running_mean,
                bn.running_var,
            ])
    for w in pre_weights:
        w.data = w[perm_map]

# modifies the weight matrix of a layer for a given permutation of the input channels
# works for both conv2d and linear
def permute_input(perm_map, after_convs):
    if not isinstance(after_convs, list):
        after_convs = [after_convs]
    post_weights = [c.weight for c in after_convs]
    for w in post_weights:
        w.data = w[:, perm_map]

def subnet(model, n_layers):
    return model.module.features[:n_layers]

def permute_m1_to_fit_m0(device, train_dset, model0, model1, batch_size=500, n_iter=99999, shuffle=True, alpha=None, alpha2=None, 
    mixup=0.0, smoothing=0.0):
    feats1 = model1.module.features
    n = len(feats1)
    loss_perm = 0
    k = 0
    for i in range(n):
        if isinstance(feats1[i], nn.Conv2d):
            # get permutation and permute output of conv and maybe bn
            if isinstance(feats1[i+1], nn.BatchNorm2d) or isinstance(feats1[i+1], nn.GroupNorm):
                assert isinstance(feats1[i+2], nn.ReLU) or isinstance(feats1[i+2], nn.Mish)
                perm_map = get_layer_perm(device, train_dset, subnet(model0, i+3), subnet(model1, i+3), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
                permute_output(perm_map, feats1[i], bn=feats1[i+1])
            else:
                assert isinstance(feats1[i+1], nn.ReLU) or isinstance(feats1[i+1], nn.Mish)
                perm_map = get_layer_perm(device, train_dset, subnet(model0, i+2), subnet(model1, i+2), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
                permute_output(perm_map, feats1[i])
            loss_perm += (perm_map.float() - torch.arange(start=0, end=perm_map.shape[0]) > 0.0).float().sum()
            k += 1

            # look for the next conv layer, whose inputs should be permuted the same way
            next_layer = None
            for j in range(i+1, n):
                if isinstance(feats1[j], nn.Conv2d):
                    next_layer = feats1[j]
                    break
            if next_layer is None:
                next_layer = model1.module.classifier
            permute_input(perm_map, next_layer)
    loss_perm = loss_perm / k
    return model1, loss_perm

## Needs to be simplified....
def permute_m1_to_fit_m0_resnet18(device, train_dset, model0, model1, batch_size=500, n_iter=99999, shuffle=True, alpha=None, alpha2=None, 
    mixup=0.0, smoothing=0.0):
    
    loss_perm = 0
    k = 0

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = self.conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.conv1, model1.module.bn1)
    permute_output(perm_map, model1.module.layer1[0].conv2, model1.module.layer1[0].bn2)
    permute_output(perm_map, model1.module.layer1[1].conv2, model1.module.layer1[1].bn2)
    permute_input(perm_map, [model1.module.layer1[0].conv1, model1.module.layer1[1].conv1,
                             model1.module.layer2[0].conv1, model1.module.layer2[0].shortcut[0]])
    loss_perm += (perm_map.float() - torch.arange(start=0, end=perm_map.shape[0]) > 0.0).float().sum()
    k += 1

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1[0].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer1[0].conv1, model1.module.layer1[0].bn1)
    permute_input(perm_map, model1.module.layer1[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1[0](x)
            x = self.layer1[1].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer1[1].conv1, model1.module.layer1[1].bn1)
    permute_input(perm_map, model1.module.layer1[1].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2[0].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer2[0].conv1, model1.module.layer2[0].bn1)
    permute_input(perm_map, model1.module.layer2[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            block = self.layer2[0]
            x = F.relu(block.bn1(block.conv1(x)))
            x = block.conv2(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer2[0].conv2, model1.module.layer2[0].bn2)
    permute_output(perm_map, model1.module.layer2[1].conv2, model1.module.layer2[1].bn2)
    permute_output(perm_map, model1.module.layer2[0].shortcut[0], model1.module.layer2[0].shortcut[1])
    permute_input(perm_map, [model1.module.layer2[1].conv1,
                             model1.module.layer3[0].conv1, model1.module.layer3[0].shortcut[0]])

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2[0](x)
            x = self.layer2[1].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer2[1].conv1, model1.module.layer2[1].bn1)
    permute_input(perm_map, model1.module.layer2[1].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3[0].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer3[0].conv1, model1.module.layer3[0].bn1)
    permute_input(perm_map, model1.module.layer3[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            block = self.layer3[0]
            x = F.relu(block.bn1(block.conv1(x)))
            x = block.conv2(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer3[0].conv2, model1.module.layer3[0].bn2)
    permute_output(perm_map, model1.module.layer3[1].conv2, model1.module.layer3[1].bn2)
    permute_output(perm_map, model1.module.layer3[0].shortcut[0], model1.module.layer3[0].shortcut[1])
    permute_input(perm_map, [model1.module.layer3[1].conv1,
                             model1.module.layer4[0].conv1, model1.module.layer4[0].shortcut[0]])

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3[0](x)
            x = self.layer3[1].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer3[1].conv1, model1.module.layer3[1].bn1)
    permute_input(perm_map, model1.module.layer3[1].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4[0].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer4[0].conv1, model1.module.layer4[0].bn1)
    permute_input(perm_map, model1.module.layer4[0].conv2)

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            block = self.layer4[0]
            x = F.relu(block.bn1(block.conv1(x)))
            x = block.conv2(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer4[0].conv2, model1.module.layer4[0].bn2)
    permute_output(perm_map, model1.module.layer4[1].conv2, model1.module.layer4[1].bn2)
    permute_output(perm_map, model1.module.layer4[0].shortcut[0], model1.module.layer4[0].shortcut[1])
    permute_input(perm_map, model1.module.layer4[1].conv1)
    model1.module.linear.weight.data = model1.module.linear.weight[:, perm_map]

    class Subnet(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            self = self.model
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4[0](x)
            x = self.layer4[1].conv1(x)
            return x
    perm_map = get_layer_perm(device, train_dset, Subnet(model0.module), Subnet(model1.module), batch_size=batch_size, n_iter=n_iter, shuffle=shuffle)
    permute_output(perm_map, model1.module.layer4[1].conv1, model1.module.layer4[1].bn1)
    permute_input(perm_map, model1.module.layer4[1].conv2)

    loss_perm = loss_perm / k
    return model1, loss_perm

     
