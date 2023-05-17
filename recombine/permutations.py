# The name of this file is not quite right, this is the code to do everything from permutation, to averaging/merging to REPAIR

import torch
import copy
from models.networks import network
from recombine.corr_matching import permute_m1_to_fit_m0, permute_m1_to_fit_m0_resnet18
from recombine.repair import correct_neuron_stats, correct_neuron_stats_multiple
from utils.utils import save_model, load_model, evaluate
from functools import partial

def mix_weights(device, net, alpha, key0, key1):
    sd0 = torch.load('%s.pt' % key0)
    sd1 = torch.load('%s.pt' % key1)
    sd_alpha = {k: (1 - alpha) * sd0[k].to(device) + alpha * sd1[k].to(device)
                for k in sd0.keys()}
    net.load_state_dict(sd_alpha)

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

# Permute -> Interpolate models -> REPAIR (optional) -> return networks
def permute_m1_to_fit_m0_with_repair(args, train_dset, test_dset, alpha, m0='vgg11_v1', m1='vgg11_v2', 
    mix=True, hyperparam=None, train_dset_noaug=None):

    permutation = args.permutation
    repair = args.repair
    batch_size = args.batch_size
    n_iter = args.n_iter
    n_iter_matching = args.n_iter_matching
    model_name = args.model_name
    shuffle = True

    model0 = network(args)
    model1 = network(args)
    load_model(model0, m0)
    load_model(model1, m1)
    if permutation:
        if 'VGG' in model_name: 
            model1, loss_perm = permute_m1_to_fit_m0(args.device, train_dset_noaug if train_dset_noaug is not None else train_dset[0], model0, model1, batch_size=batch_size, n_iter=n_iter_matching, shuffle=shuffle)
        else:
            model1, loss_perm = permute_m1_to_fit_m0_resnet18(args.device, train_dset_noaug if train_dset_noaug is not None else train_dset[0], model0, model1, batch_size=batch_size, n_iter=n_iter_matching, shuffle=shuffle)
    save_model(model1, m1) # overwrite existing model

    # Permute weights
    model0 = network(args)
    mix_weights(args.device, model0, 0.0, m0, m1)
    model1 = network(args)
    mix_weights(args.device, model1, 1.0, m0, m1)

    if not mix: # we do not mix at the last epoch
        return model0.to(args.device), model1.to(args.device)

    model_interp = network(args)
    mix_weights(args.device, model_interp, alpha, m0, m1)
    model_interp2 = network(args)
    mix_weights(args.device, model_interp2, 1-alpha, m0, m1)

    if repair: 
        model_interp = correct_neuron_stats(train_dset_noaug if train_dset_noaug is not None else train_dset[0], model0, model1, model_interp, alpha, batch_size=batch_size, 
            n_iter=n_iter, args=args, hyperparam=hyperparam)
        if alpha == 0.5:
            model_interp2 = copy.deepcopy(model_interp)
        else:
            model_interp2 = correct_neuron_stats(train_dset_noaug if train_dset_noaug is not None else train_dset[0], model0, model1, model_interp2, 1-alpha, batch_size=batch_size, 
                n_iter=n_iter, args=args, hyperparam=hyperparam)
    return model_interp.to(args.device), model_interp2.to(args.device) # replace alpha=0, alpha=1 models with alpha=.25 and alpha=.75 models

def permute_all_models_to_fit_m0_with_repair(args, train_dset, test_dset, models, model_name='VGG11',  
    mix=True, train_or_val_acc_list=None, hyperparams=None, hyperparams_after=None, train_dset_noaug=None):
    
    permutation = args.permutation
    repair = args.repair
    batch_size = args.batch_size
    n_iter = args.n_iter
    n_iter_matching = args.n_iter_matching
    model_name = args.model_name
    shuffle = True
    method_comb = args.method_comb

    perm_fn = None
    if permutation:
        if 'VGG' in model_name: 
            perm_fn = partial(permute_m1_to_fit_m0, device=args.device, train_dset=train_dset_noaug if train_dset_noaug is not None else train_dset[0], batch_size=batch_size, n_iter=n_iter_matching, shuffle=shuffle)
        else:
            perm_fn = partial(permute_m1_to_fit_m0_resnet18, device=args.device, train_dset=train_dset_noaug if train_dset_noaug is not None else train_dset[0], batch_size=batch_size, n_iter=n_iter_matching, shuffle=shuffle)

    n = models.len()
    if n == 1:
        return copy.deepcopy(models.net_list), perm_fn

    nets = copy.deepcopy(models.get_nets())
    n_nets = len(nets)
    if permutation and method_comb in ['none','many_75', 'many_half', 'random']:
        for k in range(1, n_nets):
            new_net, loss_perm = perm_fn(model0=nets[0], model1=nets[k], alpha=None)
            nets[k].load_state_dict(new_net.state_dict())

    if method_comb == 'none':
        return nets, perm_fn

    if not mix: # we do not mix at the last epoch
        return copy.deepcopy(models.net_list), perm_fn

    # Combine weights
    if method_comb == 'random':
        m = torch.distributions.dirichlet.Dirichlet(torch.ones(n, n_nets).to(args.device))
        alpha = m.sample()
    elif method_comb == 'many_75': # assuming n == n_nets
        alpha = .25/(n-1)*torch.ones(n, n).to(args.device) + (.75 - .25/(n-1))*torch.eye(n).to(args.device)
    elif method_comb == 'many_half':
        alpha = .50/(n-1)*torch.ones(n, n).to(args.device) + (.50 - .50/(n-1))*torch.eye(n).to(args.device)
    elif method_comb == 'avg':
        soup, _, alpha = models.avg_soup(train_or_val_acc_list, perm_fn=perm_fn)
    elif method_comb == 'greedy_soup':
        soup, _, alpha = models.greedy_soup(train_or_val_acc_list, perm_fn=perm_fn)
    nets_new = copy.deepcopy(models.net_list)
    for k in range(len(nets_new)):
        if method_comb == 'avg' or method_comb == 'greedy_soup':
            nets_new[k].load_state_dict(soup.state_dict())
        else:
            mix_weights_direct(args.device, alpha[k], nets_new[k], nets)
    if repair: 
        nets_new = correct_neuron_stats_multiple(train_dset, train_dset, nets, nets_new, alpha, batch_size=batch_size, 
            n_iter=n_iter, args=args, hyperparams=hyperparams, hyperparams_after=hyperparams_after)

    return nets_new, perm_fn
