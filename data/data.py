import torch
from data.cifar10 import get_cifar10
from data.cifar100 import get_cifar100
from data.imagenet import get_imagenet

def get_data(args):
    # Dataset
    if args.data == 'cifar10':
        train_dset, test_dset = get_cifar10(imagenet_size=False)
        args.num_classes = 10
        num_classes = 10
        if args.val_perc > 0:
            val_n = int(args.val_perc*len(train_dset))
            train_dset, val_dset = torch.utils.data.random_split(train_dset, [len(train_dset) - val_n, val_n])
        else:
            val_dset = None
    elif args.data == 'cifar100':
        train_dset, test_dset = get_cifar100(imagenet_size=False)
        args.num_classes = 100
        num_classes = 100
        if args.val_perc > 0:
            val_n = int(args.val_perc*len(train_dset))
            train_dset, val_dset = torch.utils.data.random_split(train_dset, [len(train_dset) - val_n, val_n])
        else:
            val_dset = None
    elif args.data == 'imagenet':
        train_dset, test_dset = get_imagenet()
        args.num_classes = 1000
        num_classes = 1000
        test_dset = torch.utils.data.Subset(test_dset, torch.randperm(len(test_dset))) # must randomize test dataset to prevent mini-batch issues
        if args.val_perc > 0:
            val_n = int(args.val_perc*len(train_dset))
            train_dset, val_dset = torch.utils.data.random_split(train_dset, [len(train_dset) - val_n, val_n])
        else:
            val_dset = None
    else:
        raise NotImplementedError('data chosen does not exists')

    return args, train_dset, test_dset, val_dset