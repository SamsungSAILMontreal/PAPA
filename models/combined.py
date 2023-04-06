import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import defaultdict
import numpy as np
import copy
import math

from models.networks import network
from utils.loss_fn import TransformTarget, SoftTargetCrossEntropy #, smooth_crossentropy
from utils.sam import SAM
from utils.optim import Optimizers
from utils.schedule import get_schedule
from utils.utils import evaluate, ensemble_evaluate, get_features, enable_running_stats, disable_running_stats
from recombine.repair import correct_neuron_stats, correct_neuron_stats_multiple

# For greedy soup, n is current number of elements in the soup
def mix_weights_soup(device, net, soup, new_model, w1, w2):
    new_soup = {k: w1 * soup[k].to(device) + w2 * new_model[k].to(device)
                for k in soup.keys()}
    net.load_state_dict(new_soup)

## List of networks
class net_list(nn.Module):
    def __init__(self, args, hyperparams, n_pop=1, 
        train_dset=None, test_dset=None, val_dset=None, 
        num_classes=10, 
        my_net_list=[], data_loaders=[], start=None):
        super(net_list, self).__init__()
        self.device = args.device
        self.args = args
        self.n_pop = n_pop
        self.hyperparams = hyperparams
        self.start = start
        self.batch_size = args.batch_size
        self.num_classes = num_classes
        self.train_dset = train_dset
        self.test_dset = test_dset
        self.val_dset = val_dset
        self.loss = SoftTargetCrossEntropy()
        self.EPOCHS = args.EPOCHS
        self.every_k_epochs = args.every_k_epochs
        self.transform_label = TransformTarget(num_classes=self.num_classes)

        if self.start is None:
            self.start = time.process_time()

        self.data_loaders = data_loaders
        if len(self.data_loaders)==0:
            for i in range(self.n_pop):
                self.data_loaders += [torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)]

        self.data_iter = []
        for i in range(self.n_pop):
            self.data_iter += [iter(self.data_loaders[i])]

        self.net_list = my_net_list
        if len(self.net_list)==0:
            for i in range(self.n_pop):
                self.net_list += [network(self.args)]
                # Make same init if required
                if args.same_init and i == 0:
                    my_state = self.net_list[i].state_dict()
                elif args.same_init and i > 0:
                    self.net_list[-1].load_state_dict(my_state)

    def get_nets(self):
        return self.net_list

    def len(self):
        return self.n_pop

    def all_parameters(self):
        params = list()
        for i in range(self.n_pop):
            params += list(self.net_list[i].parameters())
        return params

    def get_optimizers_schedulers(self):
        params = self.all_parameters()

        if 'sgd' in self.args.optim:
            optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        elif 'adamw' in self.args.optim:
            optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.wd)
        elif 'adam' in self.args.optim:
            optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.wd)
        else:
            raise NotImplementedError("optim not implemented")

        lr_scheduler = get_schedule(opt=optimizer, sched=self.args.lr_scheduler, EPOCHS=self.EPOCHS, 
            epoch_len=len(self.data_loaders[0]),
            mile=self.args.multisteplr_mile, gamma=self.args.multisteplr_gamma)

        return optimizer, lr_scheduler

    def train(self):
        for i in range(self.n_pop):
            self.net_list[i].train()

    def eval(self):
        for i in range(self.n_pop):
            self.net_list[i].eval()

    def enable_running_stats(self):
        for i in range(self.n_pop):
            enable_running_stats(self.net_list[i])

    def disable_running_stats(self):
        for i in range(self.n_pop):
            disable_running_stats(self.net_list[i])

    def replace_nets(self, nets, indexes):
        k = 0
        for j in indexes:
            self.net_list[j].load_state_dict(nets[k].state_dict())
            k += 1   
   
    # similarity between features
    def cossim(self, net=None):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        net_list = self.get_nets()
        assert self.n_pop > 1
        h1, h2, h3, h4 = get_features(self.test_dset, net_list[0], self.batch_size, maxiter=5, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers)
        mystr = ''
        if net is not None:
            net_h1, net_h2, net_h3, net_h4 = get_features(self.test_dset, net, self.batch_size, maxiter=5, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers)
            mystr += f'cos-sim net h1 = {cos(h1, net_h1).mean()}' + '\n'
            mystr += f'cos-sim net h2 = {cos(h2, net_h2).mean()}' + '\n'
            mystr += f'cos-sim net h3 = {cos(h3, net_h3).mean()}' + '\n'
            mystr += f'cos-sim net h4 = {cos(h4, net_h4).mean()}' + '\n'
            print(f'cos-sim net h1 = {cos(h1, net_h1).mean()}')
            print(f'cos-sim net h2 = {cos(h2, net_h2).mean()}')
            print(f'cos-sim net h3 = {cos(h3, net_h3).mean()}')
            print(f'cos-sim net h4 = {cos(h4, net_h4).mean()}')
        else:
            h1_, h2_, h3_, h4_ = get_features(self.test_dset, net_list[1], self.batch_size, maxiter=5, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers)
            mystr += f'cos-sim h1 = {cos(h1, h1_).mean()}' + '\n'
            mystr += f'cos-sim h2 = {cos(h2, h2_).mean()}' + '\n'
            mystr += f'cos-sim h3 = {cos(h3, h3_).mean()}' + '\n'
            mystr += f'cos-sim h4 = {cos(h4, h4_).mean()}' + '\n'
            print(f'cos-sim h1 = {cos(h1, h1_).mean()}')
            print(f'cos-sim h2 = {cos(h2, h2_).mean()}')
            print(f'cos-sim h3 = {cos(h3, h3_).mean()}')
            print(f'cos-sim h4 = {cos(h4, h4_).mean()}')
        return mystr

    def evaluate_nets(self, epoch, single_net=None, 
        soup_n=0, soup_type='', printing=True, 
        train=False, test=True, # do train, test, or neither
        maxiter=99999, test_maxiter=99999, models=None, 
        ensemble=False, loss=False):

        mystr = ''
        test_acc_list = None
        train_or_val_acc_list = None

        if ensemble:
            if models is None:
                models = self.net_list
            test_acc_list = [ensemble_evaluate(self.device, self.test_dset, models, self.batch_size, 
                maxiter=test_maxiter, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, num_classes=self.num_classes)]
            mystr += f"Epoch-{epoch+1} Ensemble test_acc = {test_acc_list[-1]} time = {time.process_time() - self.start}"
        elif single_net is not None:
            if test:
                test_acc_list = [evaluate(self.device, self.test_dset, single_net, self.batch_size, 
                    maxiter=test_maxiter, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, num_classes=self.num_classes)]
            if train:
                train_or_val_acc_list = [evaluate(self.device, self.val_dset if self.val_dset is not None else self.train_dset, single_net, self.batch_size, 
                    maxiter=maxiter, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, loss=loss, num_classes=self.num_classes)]
            if soup_type != '':
                mystr += f"{soup_type}({soup_n}) "
            mystr += f"Epoch-{epoch+1} model-{0} test_acc = {test_acc_list[-1]} time = {time.process_time() - self.start}"
        else:
            if test:
                test_acc_list = []
            if train:
                train_or_val_acc_list = []
            if models is None:
                models = self.net_list
            for j, net in enumerate(models):
                if test:
                    test_acc = evaluate(self.device, self.test_dset, net, self.batch_size, 
                        maxiter=test_maxiter, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, num_classes=self.num_classes)
                    test_acc_list += [test_acc]
                else:
                    test_acc = None
                if train:
                    train_or_val_acc = evaluate(self.device, self.val_dset if self.val_dset is not None else self.train_dset, net, self.batch_size, 
                        maxiter=maxiter, pin_memory=self.args.pin_memory, num_workers=self.args.num_workers, loss=loss, num_classes=self.num_classes)
                    train_or_val_acc_list += [train_or_val_acc]
                mystr += f"Epoch-{epoch+1} model-{j} test_acc = {test_acc} time = {time.process_time() - self.start}" + '\n'
        if printing:
            print(mystr)
        return test_acc_list, train_or_val_acc_list

    # Average models one-by-one (so that we can permute optimally with sinkhorn at every step, otherwise we would average all in one shot)
    # Note: Since we dont permute anymore its a bit redundant
    def avg_soup(self, acc_list, perm_fn=None):
        if perm_fn is None or acc_list is None:
            indexes = np.arange(0, self.n_pop)
        else:
            indexes = np.argsort(acc_list)[::-1] # order from highest accuracy to lowest
        net_list = copy.deepcopy(self.get_nets())
        soup = net_list[indexes[0]]
        n = 1
        n_pop = len(net_list)
        alpha = torch.zeros(n_pop).to(self.args.device)
        alpha[indexes[0]] = 1
        for i in indexes[1:]:
            # Adding the next model to the soup
            new_soup = network(self.args)
            if perm_fn is not None:
                perm_net, _ = perm_fn(model0=soup, model1=net_list[i], alpha=(n / (n + 1)), alpha2=(1 / (n + 1)),
                    mixup=float(self.hyperparams[i]['mixup']), smoothing=float(self.hyperparams[i]['smooth']))
            else:
                perm_net = net_list[i]
            mix_weights_soup(self.args.device, new_soup, soup.state_dict(), perm_net.state_dict(), w1=(n / (n + 1)), w2=(1 / (n + 1)))
            soup = new_soup
            n += 1
            alpha[i] = 1
        alpha /= n
        if self.args.repair_soup and n > 1:
            soup = correct_neuron_stats_multiple(self.train_dset, net_list, [soup], alpha.repeat(1, 1), batch_size=self.batch_size, 
                n_iter=self.args.n_iter, args=self.args)[0]

        return soup, n, alpha.repeat(len(self.net_list), 1)

    # Greedily try to add best model to the soup (if improves performance, we add; otherwise we go to the next one)
    def greedy_soup(self, acc_list, perm_fn=None, maxiter=99999):
        indexes = np.argsort(acc_list)[::-1] # order from highest accuracy to lowest
        net_list = copy.deepcopy(self.get_nets())
        soup = net_list[indexes[0]]
        train_or_val_acc = copy.deepcopy(acc_list[indexes[0]])
        n = 1
        alpha = torch.zeros(len(indexes)).to(self.args.device)
        alpha[indexes[0]] = 1
        for i in indexes[1:]:
            # Try adding next model to the soup
            new_soup = network(self.args)
            if perm_fn is not None:
                perm_net, _ = perm_fn(model0=soup, model1=net_list[i], alpha=(n / (n + 1)), alpha2=(1 / (n + 1)),
                    mixup=float(self.hyperparams[i]['mixup']), smoothing=float(self.hyperparams[i]['smooth']))
            else:
                perm_net = net_list[i]
            mix_weights_soup(self.args.device, new_soup, soup.state_dict(), perm_net.state_dict(), w1=(n / (n + 1)), w2=(1 / (n + 1)))
            train_or_val_acc_new = evaluate(self.device, self.val_dset if self.val_dset is not None else self.train_dset, 
                new_soup, self.batch_size, maxiter=maxiter, pin_memory=self.args.pin_memory, 
                num_workers=self.args.num_workers, loss=False, num_classes=self.num_classes)
            if train_or_val_acc_new > train_or_val_acc:
                train_or_val_acc = train_or_val_acc_new
                soup = new_soup
                n += 1
                alpha[i] = 1
        alpha /= n
        if self.args.repair_soup and n > 1:
            soup = correct_neuron_stats_multiple(self.train_dset, net_list, [soup], alpha.repeat(1, 1), batch_size=self.batch_size, 
                n_iter=self.args.n_iter, args=self.args)[0]
        return soup, n, alpha.repeat(len(self.net_list), 1)

    def forward(self):
        out = []
        labels = []
        for i in range(self.n_pop):
            try:
                x, y = next(self.data_iter[i])
            except:
                self.data_iter[i] = iter(self.data_loaders[i])
                x, y = next(self.data_iter[i])
            x, y = x.to(self.args.device), y.to(self.args.device)
            x, y_ = self.transform_label(x, y, mixup=float(self.hyperparams[i]['mixup']), smoothing=float(self.hyperparams[i]['smooth']))
            x = x.to(memory_format=torch.channels_last)
            out_ = self.net_list[i](x).unsqueeze(dim=0)
            out += [out_]
            labels += [y_]
        return torch.cat(out, dim=0), torch.cat(labels, dim=0) # npop x bs x c x h x w

    def loss_fn(self, outputs, labels):
        b_expended = outputs.shape[0]*outputs.shape[1]
        # npop x bs x c x h x w -> npop*bs x c x h x w
        loss = self.loss(outputs.view(b_expended, *outputs.shape[2:]), labels.view(b_expended, self.num_classes))
        loss = loss.view(outputs.shape[0], outputs.shape[1]) # pop x b x ...
        loss = loss.mean(dim=1).sum(dim=0) # mean over batches, sum over pop
        return loss
