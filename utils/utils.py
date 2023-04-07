'''VGG11/13/16/19 in Pytorch.'''
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.batchnorm import _BatchNorm
from utils.loss_fn import TransformTarget,  SoftTargetCrossEntropy
import os
import torch.distributed as dist
import random
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def preprocessing(rank, args, hyperparams):
    if args.world_size == 1:
        print(f'Using {torch.cuda.device_count()} GPU(s)')
        if torch.cuda.device_count() > 1:
            args.sync = True
        else:
            args.sync = False
        args.num_workers = torch.cuda.device_count()*args.num_workers
        args.device = torch.device("cuda")
        args.device_cpu = torch.device("cpu")
        args.n_pop_per_gpu = args.n_pop
        hyperparams_per_gpu = hyperparams
        residual = 0
    else:
        print(f'rank={rank}')
        setup(rank, args.world_size) # setup the process groups
        args.device = torch.device("cuda:{}".format(rank))
        args.device_cpu = torch.device("cpu:{}".format(rank))
        # make each gpu have n_pop // world_size models
        residual = args.n_pop % args.world_size
        args.n_pop_per_gpu = args.n_pop // args.world_size
        if rank == args.world_size - 1:
            args.n_pop_per_gpu += residual
        print(f'rank-{rank} {args.n_pop_per_gpu}')

        # set seeds
        torch.manual_seed(args.seed + rank + 1)
        torch.cuda.manual_seed(args.seed + rank + 1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        random.seed(args.seed + rank + 1)
        np.random.seed(args.seed + rank + 1)

        # set hyperparams
        if rank == args.world_size - 1: #0, 1, 2 for 7 models = 2 2 3 =0/1 2/3 4/5/6
            hyperparams_per_gpu = hyperparams[rank*(args.n_pop_per_gpu-residual):(rank+1)*args.n_pop_per_gpu]

        else:
            hyperparams_per_gpu = hyperparams[rank*args.n_pop_per_gpu:(rank+1)*args.n_pop_per_gpu]

    args.range_merge[0] *= args.EPOCHS
    args.range_merge[1] *= args.EPOCHS

    if args.method_comb == 'none':
        args.repair_soup = True
        print("---- REPAIR has been activated for soups ----")
        if args.model_name.lower() in ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet20']:
            args.permutation = True
            print("---- For the VGG, Resnet18, and Resnet20 architectures, we do permutation alignment for better soups since the code is available ----")
        else:
            print("---- We are NOT doing permutation alignment for better soups because the code only exists for VGG, Resnet18, and Resnet20 architectures ----")
    if args.selection:
        max_select = args.n_pop if not args.selection else int(args.n_children_per_parent*args.n_pop)
        n_groups = torch.combinations(torch.arange(0, args.n_pop).to(args.device))
        max_select = min(max_select, len(n_groups))
    else:
        max_select = args.n_pop

    if args.grad_scaler:
        scaler = GradScaler()
    else:
        scaler = None

    return args, max_select, scaler, hyperparams_per_gpu, residual

def print_args(args, hyperparams, n_hyperparams):

    if n_hyperparams == 1:
        mystr_write = '\n' + str(hyperparams[0]) + '\n'
    else:
        mystr_write = '\n' + str(hyperparams) + '\n'
    mystr_write += f'world-size={args.world_size} seed={args.seed} {args.data} {args.model_name} n_GPUs={torch.cuda.device_count()} mixed={args.mixed_precision} grad_scaler={args.grad_scaler} '
    mystr_write +=  f'n_hyper={n_hyperparams} pop={args.n_pop} comb={args.method_comb} every-{args.every_k_epochs} '
    mystr_write +=  f'epochs={args.EPOCHS}[merge=({args.range_merge[0]}, {args.range_merge[1]})] bs={args.batch_size} '
    mystr_write +=  f'opt={args.optim}[momentum={args.momentum}, clip_grad={args.clip_grad}, lr={args.lr}] lr_scheduler={args.lr_scheduler} miles={args.multisteplr_mile} gamma={args.multisteplr_gamma} '
    if args.ema_alpha != 1.0:
        mystr_write +=  f'PAPA-gradual(ema_alpha={args.ema_alpha}, every={args.ema_every_k} '
    if not args.permutation:
        mystr_write +=  f'noperm '
    else:
        mystr_write +=  f'feat-match[n_iter={args.n_iter_matching}] '
    if args.repair:
        mystr_write +=  f'repair(n_iter={args.n_iter},soup={args.repair_soup}) '
    if args.same_init:
        mystr_write +=  f'same_init '
    if not args.mix_from_start:
        mystr_write +=  f'no-mix_from_start '
    mystr_write +=  f'val={args.val_perc} '
    if args.selection:
        mystr_write +=  f'SELection(p={args.elitism_p},n={args.n_children_per_parent}, maxit={args.elitism_maxiter}) '
    if args.mutation_sigma > 0:
        mystr_write +=  f'mutate(s={args.mutation_sigma},norm={args.mutation_normalize}) '
    mystr_write += '\n'
    print(mystr_write)
    return mystr_write

def save_model(model, i):
    torch.save(model.state_dict(), '%s.pt' % i)

def load_model(model, i):
    sd = torch.load('%s.pt' % i)
    model.load_state_dict(sd)

def same_model(model1, model2):
    for param, param2 in zip(model1.parameters(), model2.parameters()): 
        if ((param - param2) ** 2).sum() > 0:
            return False
    return True

def get_features(device, test_dset, model, batch_size=500, maxiter=99999, pin_memory=False, num_workers=0):
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    model.eval()
    with torch.no_grad(), autocast():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= maxiter:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(memory_format=torch.channels_last)
            if i == 0:
                h1, h2, h3, h4 = model.module.forward_layers(inputs)
            else:
                h1_, h2_, h3_, h4_ = model.module.forward_layers(inputs)
                h1 = torch.cat([h1, h1_], dim=0)
                h2 = torch.cat([h2, h2_], dim=0)
                h3 = torch.cat([h3, h3_], dim=0)
                h4 = torch.cat([h4, h4_], dim=0)
    model.train()
    return h1, h2, h3, h4

def evaluate(device, test_dset, model, batch_size=500, maxiter=99999, pin_memory=False, num_workers=0, loss=False, num_classes=100, printing=False):
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    model.eval()
    correct = 0
    my_loss = 0
    len_end = len(test_loader.dataset)
    loss_fn = SoftTargetCrossEntropy()
    transform_label = TransformTarget(num_classes=num_classes)
    with torch.no_grad(), autocast():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= maxiter:
                len_end = batch_size*i
                break
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(memory_format=torch.channels_last)
            inputs, labels_ = transform_label(inputs, labels, mixup=0.0, smoothing=0.0)
            outputs = model(inputs)
            if loss:
                my_loss += loss_fn(outputs, labels_).sum().item()
            else:
                pred = outputs.argmax(dim=1)
                correct += (labels == pred).sum().item()
    model.train()
    if loss:
        return my_loss / len_end
    else:
        return correct / len_end

# evaluate using logit-averaging ensembling
def ensemble_evaluate(device, test_dset, models, batch_size=500, maxiter=99999, pin_memory=False, num_workers=0, loss=False, num_classes=100):
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    for model in models:
        model.eval()
    correct = 0
    my_loss = 0
    len_end = len(test_loader.dataset)
    loss_fn = SoftTargetCrossEntropy()
    transform_label = TransformTarget(num_classes=num_classes)
    with torch.no_grad(), autocast():
        for i, (inputs, labels) in enumerate(test_loader):
            if i >= maxiter:
                len_end = batch_size*i
                break
            outputs = 0
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(memory_format=torch.channels_last)
            inputs, labels_ = transform_label(inputs, labels, mixup=0.0, smoothing=0.0)
            for model in models:
                outputs += model(inputs)/len(models)
            if loss:
                my_loss += loss_fn(outputs, labels_).sum().item()
            else:
                pred = outputs.argmax(dim=1)
                correct += (labels == pred).sum().item()
    for model in models:
        model.train()
    if loss:
        return my_loss / len_end
    else:
        return correct / len_end

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)

def mutate_model(device, models, sigma=0, normalize=False):
    for model in models:
        with torch.no_grad():
            for param in model.parameters():
                if sum([*param.data.shape]) > 1:
                    if normalize:
                        param_norm = torch.norm(param.data,p=2)
                    else:
                        param_norm = 1
                    param.data.add_(torch.randn(param.data.size()).to(device) * sigma / param_norm)
    return models
