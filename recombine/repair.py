## Implementation of REPAIR, heavily modified from their original code (https://github.com/KellerJordan/REPAIR)

#### Generalizing this code to other architectures
# This version assumes that one use 2d convolutional networks.
# If you want to use this for 1d Conv, 3d Conv, or even MLP setting, you need to change all "isinstance(module, torch.nn.Conv2d)" to "isinstance(module, BLABLA)".
# Note that using "isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)" could also be useful (but not required since REPAIR doesn't have to be applied everywhere) if you use a complex network which sometimes use dense layers
# Furthermore, if you are in 1d or 3d; you will have to change the nn.BatchNorm2d to other types of batch norm layers for 1d or 3d setting
# So as you can see, generalizing this code to non 2d-convolutional networks is not very difficult

import copy
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from utils.loss_fn import TransformTarget
import numpy as np

class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)
        
    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())
        
    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1

class ResetLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)
        
    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std
        
    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)

# adds TrackLayers around every conv layer
def make_tracked_net(net, args):
    net1 = copy.deepcopy(net)
    n = len(list(net1.modules()))
    for i, (name, module) in enumerate(net.named_modules()): 
        if isinstance(module, torch.nn.Conv2d) and i != (n - 1):
            # Get to before-last module and then set attr to new module (necessary annoying trick, just replacing does not work)
            tokens = name.strip().split('.')
            layer = net1
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            setattr(layer, tokens[-1], TrackLayer(module))
    return net1.to(args.device).eval()

# adds ResetLayers around every conv layer
def make_repaired_net(net, args):
    net1 = copy.deepcopy(net)
    n = len(list(net1.modules()))
    for i, (name, module) in enumerate(net.named_modules()): 
        if isinstance(module, torch.nn.Conv2d) and i != (n - 1):
            # Get to before-last module and then set attr to new module (necessary annoying trick, just replacing does not work)
            tokens = name.strip().split('.')
            layer = net1
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            setattr(layer, tokens[-1], ResetLayer(module))
    return net1.to(args.device).eval()

# reset all tracked BN stats against training data
def reset_bn_stats(device, train_dset, model, batch_size=500, n_iter=99999, seed=None, cuda_seed=None, mixup=0.0, smoothing=0.0, cutmix=0.0, transform=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(cuda_seed)
    train_aug_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=0)
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None # use simple average
            m.reset_running_stats()
    model.train()
    with torch.no_grad(), autocast():
        for i, (x, y) in enumerate(train_aug_loader):
            if i >= n_iter:
                break
            x, y = x.to(device), y.to(device)
            if transform is not None:
                x, y = transform(x, y, mixup=mixup, smoothing=smoothing, cutmix=cutmix)
            x = x.to(memory_format=torch.channels_last)
            output = model(x)

def correct_neuron_stats(train_dset, model0, model1, model_interp, alpha, batch_size=500, n_iter=99999, 
    args=None, hyperparam=None, same_repair=True):
    
    mixup = 0.0
    smoothing = 0.0
    cutmix = 0.0
    if hyperparam is not None:
        mixup = float(hyperparam['mixup'])
        smoothing = float(hyperparam['smooth'])
        cutmix = float(hyperparam['cutmix'])
    if same_repair: # option I added, ensure same data each time, which makes much more sense and improve results slightly
        seed = np.random.randint(99999)
        cuda_seed = np.random.randint(99999)
    else:
        seed = None
        cuda_seed = None
    transform = TransformTarget(args.num_classes)

    wrap0 = make_tracked_net(model0, args=args)
    wrap1 = make_tracked_net(model1, args=args)
    reset_bn_stats(args.device, train_dset, wrap0, batch_size=batch_size, n_iter=n_iter, seed=seed, cuda_seed=cuda_seed, mixup=mixup, smoothing=smoothing, cutmix=cutmix, transform=transform)
    reset_bn_stats(args.device, train_dset, wrap1, batch_size=batch_size, n_iter=n_iter, seed=seed, cuda_seed=cuda_seed, mixup=mixup, smoothing=smoothing, cutmix=cutmix, transform=transform)
    wrap_a = make_repaired_net(model_interp, args=args)
    # Iterate through corresponding triples of (TrackLayer, TrackLayer, ResetLayer)
    # around conv layers in (model0, model1, model_interp).
    for track0, track1, reset_a in zip(wrap0.modules(), wrap1.modules(), wrap_a.modules()): 
        if not isinstance(track0, TrackLayer):
            continue  
        assert (isinstance(track0, TrackLayer)
              and isinstance(track1, TrackLayer)
              and isinstance(reset_a, ResetLayer))

        # get neuronal statistics of original networks
        mu0, std0 = track0.get_stats()
        mu1, std1 = track1.get_stats()
        # set the goal neuronal statistics for the merged network 
        goal_mean = (1 - alpha) * mu0 + alpha * mu1
        goal_std = (1 - alpha) * std0 + alpha * std1
        reset_a.set_stats(goal_mean, goal_std)

    # Estimate mean/vars such that when added BNs are set to eval mode,
    # neuronal stats will be goal_mean and goal_std.
    reset_bn_stats(args.device, train_dset, wrap_a, batch_size=batch_size, n_iter=n_iter, seed=seed, cuda_seed=cuda_seed, mixup=mixup, smoothing=smoothing, cutmix=cutmix, transform=transform)
    # fuse the rescaling+shift coefficients back into conv layers
    model_interp = fuse_tracked_net(wrap_a, args=args)

    return model_interp

## Generalization to more than 2 networks combined
# O(ln^2), l=number of layers, n=number of networks
def correct_neuron_stats_multiple(train_dset_in, train_dset_out, models, models_interp, alpha, batch_size=500, 
    n_iter=99999, args=None, same_repair=False, hyperparams=None, hyperparams_after=None):

    mixup = [0.0 for i in range(len(models))]
    smoothing = [0.0 for i in range(len(models))]
    cutmix = [0.0 for i in range(len(models))]
    mixup_after = mixup
    smoothing_after = smoothing
    cutmix_after = cutmix
    if hyperparams is not None:
        mixup = [float(hyperparams[i]['mixup']) for i in range(len(models))]
        smoothing = [float(hyperparams[i]['smooth']) for i in range(len(models))]
        cutmix = [float(hyperparams[i]['cutmix']) for i in range(len(models))]
    if hyperparams_after is not None:
        mixup_after = [float(hyperparams_after[i]['mixup']) for i in range(len(models))]
        smoothing_after = [float(hyperparams_after[i]['smooth']) for i in range(len(models))]
        cutmix_after = [float(hyperparams_after[i]['cutmix']) for i in range(len(models))]

    if same_repair: # option I added, ensure same data each time, which makes much more sense and improve results slightly
        seed = np.random.randint(99999)
        cuda_seed = np.random.randint(99999)
    else:
        seed = None
        cuda_seed = None
    transform = TransformTarget(args.num_classes)
    wrap_interp = copy.deepcopy(models_interp)

    # Make tracked and repair networks
    for i in range(len(models_interp)):
        wrap_interp[i] = make_repaired_net(models_interp[i], args=args)
    wrap = copy.deepcopy(models)
    for j in range(len(models)): 
        wrap[j] = make_tracked_net(models[j], args=args)
        reset_bn_stats(args.device, train_dset_in[0 if len(train_dset_in)==1 else j], wrap[j], batch_size=batch_size, n_iter=n_iter, seed=seed, cuda_seed=cuda_seed, mixup=mixup[j], smoothing=smoothing[j], cutmix=cutmix[j], transform=transform)

    for i in range(len(models_interp)): # loop over interpolated (child) models
        goal_mean = dict()
        goal_std = dict()
        for j in range(len(models)): # loop over (parent) models
            for l, layer in enumerate(wrap[j].modules()): # loop over layers to get goal_mean, goal_std
                if isinstance(layer, TrackLayer):
                    # get neuronal statistics of original networks
                    mu, std = layer.get_stats()
                    # set the goal neuronal statistics for the merged network 
                    if l in goal_mean:
                        goal_mean[l] += alpha[i][j] * mu
                        goal_std[l] += alpha[i][j] * std
                    else:
                        goal_mean[l] = alpha[i][j] * mu
                        goal_std[l] = alpha[i][j] * std

        for l, layer in enumerate(wrap_interp[i].modules()): # loop over layers of interpolated net to fix mean and std
            if isinstance(layer, ResetLayer):
                layer.set_stats(goal_mean[l], goal_std[l])

        # Estimate mean/vars such that when added BNs are set to eval mode,
        # neuronal stats will be goal_mean and goal_std.
        reset_bn_stats(args.device, train_dset_out[0 if len(train_dset_out)==1 else i], wrap_interp[i], batch_size=batch_size, n_iter=n_iter, seed=seed, cuda_seed=cuda_seed, mixup=mixup_after[i], smoothing=smoothing_after[i], cutmix=cutmix_after[i], transform=transform)

        # fuse the rescaling+shift coefficients back into conv layers
        wrap_interp[i] = fuse_tracked_net(wrap_interp[i], args=args)

    return wrap_interp


def fuse_conv_bn(conv, bn):
    has_bias = False if conv.bias is None else True
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 bias=has_bias)

    # set weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused_conv.weight.data = (w_conv * gamma.reshape(-1, 1, 1, 1))

    # set bias
    if has_bias:
        beta = bn.bias + gamma * (-bn.running_mean + conv.bias)
        fused_conv.bias.data = beta
    
    return fused_conv

def fuse_tracked_net(net, args):
    net1 = copy.deepcopy(net)
    n = len(list(net1.modules()))
    for i, (name, rlayer) in enumerate(net1.named_modules()): 
        if isinstance(rlayer, ResetLayer):
            fused_conv = fuse_conv_bn(rlayer.layer, rlayer.bn)
            # Get to before-last module and then set attr to new module (necessary annoying trick, just replacing does not work)
            tokens = name.strip().split('.')
            layer = net1
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            setattr(layer, tokens[-1], fused_conv)
    return net1.to(args.device).eval()



