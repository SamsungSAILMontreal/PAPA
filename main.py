import argparse
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
import math
import random
import time
import copy
import os
import torch.distributed as dist
import torch.multiprocessing as mp

from models.combined import net_list
from models.networks import network, copy_networks
from recombine.permutations import permute_m1_to_fit_m0_with_repair, permute_all_models_to_fit_m0_with_repair, mix_weights_direct
from utils.utils import preprocessing, print_args, save_model, load_model, evaluate, mutate_model
from data.data import get_data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_file", type=str, default='/scratch/jolicoea/recombine_nets/final_results.txt', help="where to save the results")
    parser.add_argument("--EPOCHS", type=int, default=300, help="number of epochs")
    parser.add_argument("--n_pop", type=int, default=2, help="number of networks")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    
    # Model
    parser.add_argument("--model_name", type=str, default='VGG11', help='VGG11, VGG13, VGG16, VGG19, ResNet18, Resnet20, Resnet50 (imagenet)')
    
    # Data
    parser.add_argument("--data", type=str, default='cifar10', help='cifar10, cifar100, imagenet')
    parser.add_argument("--val_perc", type=float, default=0.0, help="If 0, soups use test data, otherwise use an hold-out of images from the training data as validation data (in the paper we did not use that option, but it would be a very sensible to do that in practice if you do not have a seperate validation data; this will lead to much better soups)")
    parser.add_argument("--pin_memory", type=str2bool, default=True, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers per gpu')

    # Mixed-precision (faster, less memory = win win)
    parser.add_argument("--mixed_precision", type=str2bool, default=True, help='If True, use mixed precision')
    parser.add_argument("--grad_scaler", type=str2bool, default=True, help='If True, scale step for mixed precision to prevent problems in gradients')

    # ESGD
    parser.add_argument("--tournament_pairwise", type=str2bool, default=False, help="If True, tournament selection")
    parser.add_argument("--selection", type=str2bool, default=False, help="If True, use elitism selection of children; otherwise do not")
    parser.add_argument("--elitism_p", type=float, default=0.60, help="Percentage 60 percent best are select, then the rest is randomly picked")
    parser.add_argument("--n_children_per_parent", type=int, default=6, help="Make 6 times more babies than parents")
    parser.add_argument("--elitism_maxiter", type=float, default=99999, help="Maxiter of evaluation of each net")
    parser.add_argument("--mutation_sigma", type=float, default=0.0, help="mutation is N(0, sigma); use 0.01")
    parser.add_argument("--mutation_normalize", type=str2bool, default=False, help="normalize the mutation")

    # Permutation alignment
    parser.add_argument("--permutation", type=str2bool, default=False, help="Permute weights using feature matching (or weight matching) before interpolating")
    parser.add_argument("--n_iter_matching", type=int, default=9999, help="maximum number of dataloader iterations for correlation/weight matching")
    
    # PAPA (method_comb=pair_half is PAPA-2, method_comb=avg is PAPA-all)
    parser.add_argument("--every_k_epochs", type=int, default=5, help="apply Genetic algorithm every k epochs")
    parser.add_argument("--method_comb", type=str, default='avg', help=" Weights for combining networks \
        No merging: none [Baseline], \
        Combine pairs of 2 nets: pair_75 (.75, .25), pair_half (.5, .5) [PAPA-2], \
        Combine all nets: many_half (.50, .50*1/k, ... , .50*1/k), many_75 (.75, .25*1/k, ... , .25*1/k), \
        Average of all nets duplicated over all children: avg (1/k, ... , 1/k) [PAPA-all],  \
        Soup (duplicated over all children): greedy_soup (avg of best nets) \
        Combine all nets with random weights: random (random1, random2, ... , randomk)")
    parser.add_argument("--same_init", type=str2bool, default=False, help="If True, all nets start from the same initialization")
    parser.add_argument("--mix_from_start", type=str2bool, default=True, help="If True, mix the networks at the end of epoch 0")
    parser.add_argument('--range_merge', nargs=2, type=float, default=[0,1], help='Range of timing of which to use the network merging (ex: [0.2,.5] with 100 epochs will only merge networks between epoch 200 and epoch 500; inclusive)')

    # PAPA-gradual
    parser.add_argument("--ema_alpha", type=float, default=1.0, help="ema_alpha*w + (1-ema_alpha)*w_avg, using ema_alpha < 1.0 will disable any option used in method_comb and PAPA")
    parser.add_argument("--ema_every_k", type=int, default=1, help="how often (in iteration, not epoch) to apply the EMA, default to every single iteration")

    # Optimization
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizers to choose from (adam, sgd, adamw)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument("--clip_grad", type=float, default=0, help='clip grad value')
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd momentum parameter")
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='multisteplr', help='none, multisteplr')
    parser.add_argument('--multisteplr_mile', type=float, nargs='+', default=[.5, .75], help='milestones for lr reductions')
    parser.add_argument('--multisteplr_gamma', type=float, default=0.1, help='gamma for lr reductions')

    # REPAIR
    parser.add_argument("--repair", type=str2bool, default=True, help="use REPAIR after interpolating")
    parser.add_argument("--repair_soup", type=str2bool, default=False, help="use REPAIR after soups (will be set to True for method_comb=none)")  # bad, but needed for baseline models to merge well in soups
    parser.add_argument("--n_iter", type=int, default=5, help="the number of itererations in repair")

    # Varying Regularizations
    parser.add_argument('--mixup', nargs='+', default=['0.0'], help='List of mixup alpha to choose from')
    parser.add_argument('--smooth', nargs='+', default=['0.0'], help='List of label smoothings to choose from')
    parser.add_argument("--hyperparams_mix", type=str2bool, default=True, help='If true, can have label smoothing and mixup together, otherwise its one or the')

    # Correlation estimation
    parser.add_argument("--correlation_est", type=str2bool, default=False, help='Calculate the cosine similarity between layers of all networks to the avg-soup')

    # Debug
    parser.add_argument("--test", type=str2bool, default=False, help='If true, one-training step only')

    # Multiprocessing to parallelize over the population
    parser.add_argument('--world_size', type=int, default=1, help='If bigger than 1 uses distributed learning with <world_size> GPUs; \
        e.g., if n_pop=10 and world-size=3, GPU-0 will do the first 3 models, GPU-1 the next 3 models, and GPU-2 the last 4 models. \
        This option assumes 1 GPU per process, so it will not work if your model-size or batch-size too large to be handled by a single GPU. \
        Generalization to multiple GPUs per process would be super useful, but I am not sure how to do it. \
        Note that GPU-0 must be able to fit all the networks of the population inside it because GPU-0 will do the averaging/evolution stage.')

    args = parser.parse_args()
    return args

def main(rank, args, hyperparams, n_hyperparams):
    
    # Preprocessing of args
    args, max_select, scaler, hyperparams_per_gpu, residual = preprocessing(rank, args, hyperparams)

    # Dataset
    args, train_dset, test_dset, val_dset = get_data(args)

    # print args
    mystr_write = print_args(args, hyperparams_per_gpu, n_hyperparams)

    # List of networks
    models = net_list(args=args, n_pop=args.n_pop_per_gpu, hyperparams=hyperparams_per_gpu, 
        train_dset=train_dset, test_dset=test_dset, val_dset=val_dset, num_classes=args.num_classes)
    optimizer, lr_scheduler = models.get_optimizers_schedulers()

    perm_fn = None
    len_data_loader = len(models.data_loaders[0])
    start = time.process_time()
    n_generation = 0
    if args.ema_alpha != 1.0:
        ema_alpha = torch.tensor([args.ema_alpha, 1-args.ema_alpha], device=args.device)
        net_avg = copy.deepcopy(models.get_nets()[0])
        unif_weights = torch.ones(args.n_pop, device=args.device)/args.n_pop
    perm_ = torch.arange(0, models.len()).to(args.device)

    if args.world_size > 1:
        dist.barrier() # probably not needed but its nice to wait for everyone :)

    for epoch in range(args.EPOCHS):
        models.train()
        ## Train for one epoch
        for i in range(len_data_loader):
            optimizer.zero_grad()
            if args.mixed_precision:
                with autocast():
                    outputs, labels = models()
                    loss = models.loss_fn(outputs, labels)
            else:
                outputs, labels = models()
                loss = models.loss_fn(outputs, labels)
            #print(f'epoch-{epoch} step-{i} loss={loss.data}')
            if args.grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(models.all_parameters(), args.clip_grad, norm_type=2.0)
            if args.grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()
            if args.test:
                break #### FOR DEBUGGING

            # EMA of current nets and the average
            if args.ema_alpha != 1.0 and (i % args.ema_every_k == 0):
                nets = models.get_nets()
                # make current avg net
                mix_weights_direct(args.device, unif_weights, net_avg, nets)
                # EMA of current nets and the average
                for k in range(args.n_pop):
                    mix_weights_direct(args.device, ema_alpha, nets[k], [nets[k], net_avg])

        # Evaluate networks

        # PAPA - merging
        if args.ema_alpha == 1.0 and ((args.method_comb == 'none' and epoch == args.EPOCHS - 1) or ((not args.method_comb == 'none') and (epoch % args.every_k_epochs == 0 and (epoch > 0 or args.mix_from_start) and (epoch+1 >= args.range_merge[0] and epoch+1 <= args.range_merge[1])))):
            n_generation += 1

            if args.n_pop > 1:
                # Distributed must set up the world net_list
                if args.world_size > 1:
                    torch.save({'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}, f'opt_{rank}.pt') # backup optimizer, scheduler
                    # Gather nets and loaders
                    net_list_world = [None for _ in range(args.world_size)]
                    data_loaders_world = [None for _ in range(args.world_size)]
                    dist.barrier()
                    dist.gather_object(models.net_list, net_list_world if rank == 0 else None, dst=0)
                    dist.gather_object(models.data_loaders, data_loaders_world if rank == 0 else None, dst=0)
                    del models
                    del optimizer
                    torch.cuda.empty_cache()

                # GPU does everything if distributed
                if rank == 0:
                    # GPU-0 must remake a net list
                    if args.world_size > 1:
                        net_list_0 = []
                        data_loaders_0 = []
                        for i in range(args.world_size):
                            net_list_0 += net_list_world[i]
                            data_loaders_0 += data_loaders_world[i]
                        for i in range(args.n_pop):
                            net_list_0[i] = net_list_0[i].to(args.device)
                        models = net_list(args=args, n_pop=args.n_pop, hyperparams=hyperparams, 
                                train_dset=train_dset, test_dset=test_dset, val_dset=val_dset, num_classes=args.num_classes,
                                my_net_list=net_list_0, data_loaders=data_loaders_0, start=start).to(args.device)

                        # Bring back nets and loaders to processes
                        net_list_world = [None for i in range(args.world_size)]
                        data_loaders_world = [None for i in range(args.world_size)]
                        k = 0
                        for i in range(args.world_size):
                            n_pop_per_gpu_ = args.n_pop // args.world_size
                            if i == args.world_size - 1:
                                n_pop_per_gpu_ += residual
                            net_list_world[i] = copy_networks(args, models.net_list[k:(k+n_pop_per_gpu_)])
                            data_loaders_world[i] = copy.deepcopy(models.data_loaders[k:(k+n_pop_per_gpu_)])
                            k += n_pop_per_gpu_

                    # Permute (optional), merge, REPAIR (optional)
                    #test_acc_list, train_or_val_acc_list = models.evaluate_nets(epoch, train=False, test=True, printing=True) ### REMOVE, FOR DEBUGGING
                    if args.method_comb == 'greedy_soup':
                        _, train_or_val_acc_list = models.evaluate_nets(epoch, train=True, printing=False, test=False)
                    else:
                        train_or_val_acc_list = None
                    if args.method_comb in ['none','many_75', 'many_half', 'avg','random', 'greedy_soup']:
                        rand_perm = torch.arange(0, models.len()).to(args.device) # not random actually, just 0, 1, ... , n
                        new_models, perm_fn = permute_all_models_to_fit_m0_with_repair(args=args, 
                                    train_dset=train_dset, test_dset=test_dset, models=models,
                                    mix=epoch < args.EPOCHS - 1,
                                    train_or_val_acc_list=train_or_val_acc_list, 
                                    hyperparams=hyperparams)
                    else:
                        for k, model_k in enumerate(models.get_nets()):
                            save_model(model_k, args.model_name+'_'+str(k))

                        rand_perm = torch.arange(0, models.len()).to(args.device)

                        if args.tournament_pairwise and args.n_pop >= 3: # Tournament selection
                            _, train_or_val_loss_list = models.evaluate_nets(epoch, single_net=None, train=True, test=False, printing=False, loss=True, maxiter=args.elitism_maxiter)
                            # 1/loss_1, 1/loss_2 => normalized
                            train_or_val_loss_list_inv = [1/(1e-05 + xxxxx) for xxxxx in train_or_val_loss_list]
                            train_or_val_loss_list_inv_p = [xxxxx/sum(train_or_val_loss_list_inv) for xxxxx in train_or_val_loss_list_inv]
                            n_groups = torch.zeros(max_select, 2, dtype=torch.int).to(args.device)
                            for zk in range(max_select):
                                rand_choices = np.random.choice(args.n_pop, 2, replace=False, p=train_or_val_loss_list_inv_p)
                                n_groups[zk][0] = rand_choices[0]
                                n_groups[zk][1] = rand_choices[1]
                        elif args.n_pop >= 3:
                            # Get all pairwise combinations and select n_pop random pairs
                            n_groups = torch.combinations(torch.arange(0, models.len()).to(args.device))
                            n_groups = n_groups[torch.randperm(max_select).to(args.device)][:max_select]
                            # In this setting, rand_perm has zero impact, so we merge random pair and place the nets into random environments
                        else: # if args.n_pop == 2
                            n_groups = torch.tensor([[0,1],[0,1]])
                        print(f'groups=[{n_groups}]')

                        if epoch == args.EPOCHS - 1: # at last epoch, we do not merge, we only permute (permute is needed for greedy model soup)
                            u = torch.zeros(len(n_groups), dtype=torch.float32).to(args.device)
                        else:
                            u = torch.ones(len(n_groups), dtype=torch.float32).to(args.device)
                            if args.method_comb == 'pair_75': # .25 / .75
                                u = u * .25
                            elif args.method_comb == 'pair_half': # .50 / .50
                                u = u * .5
                        new_models = []
                        for j, indexes in enumerate(n_groups):
                            new_models += [permute_m1_to_fit_m0_with_repair(args=args, train_dset=train_dset, test_dset=test_dset, alpha=u[j], 
                                m0=args.model_name+'_'+str(indexes[0].item()), m1=args.model_name+'_'+str(indexes[1].item()), 
                                mix=epoch < args.EPOCHS - 1)[random.randint(0, 1)]]
                    
                    # Mutate these babies to make monstrosities
                    if args.mutation_sigma > 0:
                        new_models = mutate_model(args.device, new_models, sigma=args.mutation_sigma / n_generation, normalize=args.mutation_normalize)
                    
                    # Drop non-winner in a elitism selection (ESGD)
                    if args.selection:
                        new_models = new_models + copy_networks(args, models.get_nets())
                        _, train_or_val_loss_list = models.evaluate_nets(epoch, single_net=None, train=True, test=False, loss=True, models=new_models, printing=False, maxiter=args.elitism_maxiter)
                        indexes_loss = np.argsort(train_or_val_loss_list) # order from lowest loss to highest
                        n_elite = int(args.n_pop*args.elitism_p)
                        survivors = []
                        # Choose who survives
                        for i in range(n_elite):
                            survivor_i = new_models[indexes_loss[i]]
                            survivors += [survivor_i]
                        # The rest gambles to survive :'(
                        rand_survivors = n_elite + torch.randperm(args.n_pop - n_elite).to(args.device)
                        for i in range(args.n_pop - n_elite):
                            survivor_i = new_models[indexes_loss[rand_survivors[i]]]
                            survivors += [survivor_i]
                        assert len(survivors) == args.n_pop
                        new_models = survivors

                    # Update models
                    models.replace_nets(new_models, rand_perm)
                    #test_acc_list, train_or_val_acc_list = models.evaluate_nets(epoch, train=False, test=True, printing=True) ### REMOVE, FOR DEBUGGING
                    del new_models

                    if args.world_size > 1:
                        # Bring back evaluate nets and loaders to processes/GPUs
                        net_list_world = [None for i in range(args.world_size)]
                        data_loaders_world = [None for i in range(args.world_size)]
                        k = 0
                        for i in range(args.world_size):
                            n_pop_per_gpu_ = args.n_pop // args.world_size
                            if i == args.world_size - 1:
                                n_pop_per_gpu_ += residual
                            net_list_world[i] = copy_networks(args, models.net_list[k:(k+n_pop_per_gpu_)])
                            data_loaders_world[i] = copy.deepcopy(models.data_loaders[k:(k+n_pop_per_gpu_)])
                            k += n_pop_per_gpu_

                else:
                    net_list_world = [None for i in range(args.world_size)]
                    data_loaders_world = [None for i in range(args.world_size)]
                if args.world_size > 1:
                    net_list_ = [None]
                    data_loaders_ = [None]
                    torch.cuda.empty_cache()
                    dist.barrier()
                    dist.scatter_object_list(data_loaders_, data_loaders_world, src=0)
                    dist.scatter_object_list(net_list_, net_list_world, src=0)
                    # Redo net list and optimizer
                    for i in range(args.n_pop_per_gpu):
                        net_list_[0][i] = net_list_[0][i].to(args.device)
                    models = net_list(args=args, n_pop=args.n_pop_per_gpu, hyperparams=hyperparams_per_gpu,
                            train_dset=train_dset, test_dset=test_dset, val_dset=val_dset, num_classes=args.num_classes, 
                            my_net_list=net_list_[0], data_loaders=data_loaders_[0], start=start).to(args.device)
                    optimizer, lr_scheduler = models.get_optimizers_schedulers()
                    dist.barrier()
                    # Resume optimizer, scheduler
                    checkpoint = torch.load(f'opt_{rank}.pt')
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # GPU-0 must remake a net list
    if args.world_size > 1:

        # Gather nets and loaders
        net_list_world = [None for _ in range(args.world_size)]
        data_loaders_world = [None for _ in range(args.world_size)]
        dist.barrier()
        dist.gather_object(models.net_list, net_list_world if rank == 0 else None, dst=0)
        dist.gather_object(models.data_loaders, data_loaders_world if rank == 0 else None, dst=0)
        del models
        del optimizer
        torch.cuda.empty_cache()

    if rank == 0:

        if args.world_size > 1:
            net_list_0 = []
            data_loaders_0 = []
            for i in range(args.world_size):
                net_list_0 += net_list_world[i]
                data_loaders_0 += data_loaders_world[i]
            for i in range(args.n_pop):
                net_list_0[i] = net_list_0[i].to(args.device)
            models = net_list(args=args, n_pop=args.n_pop, hyperparams=hyperparams, 
                    train_dset=train_dset, test_dset=test_dset, val_dset=val_dset, num_classes=args.num_classes,
                    my_net_list=net_list_0, data_loaders=data_loaders_0, start=start).to(args.device)

        test_acc_list, train_or_val_acc_list = models.evaluate_nets(epoch, train=True, test=True, ensemble=False)
        ensemble_acc_list, _ = models.evaluate_nets(epoch, train=False, test=False, ensemble=True)

        # Get soups at the end and evaluate them
        if args.correlation_est:
            cossim_str1 = models.cossim()
        soup, asoup_n, _ = models.avg_soup(train_or_val_acc_list, perm_fn=perm_fn)
        if args.correlation_est:
            cossim_str2 = models.cossim(soup)
        acc_asoup, _ = models.evaluate_nets(epoch, single_net=soup, soup_n=asoup_n, soup_type="AvgSoup")
        soup, gsoup_n, _ = models.greedy_soup(train_or_val_acc_list, perm_fn=perm_fn)
        acc_gsoup, _ = models.evaluate_nets(epoch, single_net=soup, soup_n=gsoup_n, soup_type="GreedySoup")

        print(f'\n')
        last_acc_tensor = torch.tensor(test_acc_list, dtype=torch.float)
        ensemble_acc_tensor = torch.tensor(ensemble_acc_list, dtype=torch.float)
        mystr_write += f"Models Mean [min, max] & Ens & AvgSoup & GreedySoup = {round(last_acc_tensor.mean().item()*100, 2)} [{round(last_acc_tensor.min().item()*100, 2)}, {round(last_acc_tensor.max().item()*100, 2)}] & {round(ensemble_acc_tensor.item()*100, 2)} & {round(acc_asoup[0]*100, 2)} & {round(acc_gsoup[0]*100, 2)}" + '\n'
        if args.correlation_est:
            mystr_write += cossim_str1
            mystr_write += cossim_str2
        print(mystr_write)
        with open(args.results_file, 'a') as f: # where we keep track of the results
            f.write(mystr_write)

    if args.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    
    args = load_config()

    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Choose the hyperparameters to be used
    hyperparams = []
    if args.hyperparams_mix:
        for m in range(len(args.mixup)):
            for n in range(len(args.smooth)):
                hyperparams += [{'mixup': args.mixup[m], 'smooth': args.smooth[n]}]
    else:
        hyperparams += [{'mixup': args.mixup[0], 'smooth': args.smooth[0]}]
        for i in range(1, len(args.mixup)):
            hyperparams += [{'mixup': args.mixup[i], 'smooth': args.smooth[0]}]
        for i in range(1, len(args.smooth)):
            hyperparams += [{'mixup': args.mixup[0], 'smooth': args.smooth[i]}]


    n_hyperparams = len(hyperparams)
    random.shuffle(hyperparams)
    assert args.n_pop > 0 and n_hyperparams > 0
    n_repeat = -(-args.n_pop // n_hyperparams)
    hyperparams = [hyperparams[i % n_hyperparams] for i in range(args.n_pop)]

    if args.world_size > 1:
        mp.spawn(main, args=(args, hyperparams, n_hyperparams), nprocs=args.world_size)
    else:
        main(0, args, hyperparams, n_hyperparams)
