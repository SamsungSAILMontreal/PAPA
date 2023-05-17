import torch
from data.cifar10 import get_cifar10
from data.cifar100 import get_cifar100
from data.imagenet import get_imagenet

def get_data(args, hyperparams):
    # Dataset

    if args.data == 'cifar10':
        data_fun = get_cifar10
        args.num_classes = num_classes = 10
    elif args.data == 'cifar100':
        data_fun = get_cifar100
        args.num_classes = num_classes = 100
    elif args.data == 'imagenet':
        data_fun = get_imagenet
        args.num_classes = num_classes = 1000
    else:
        raise NotImplementedError('data chosen does not exists')

    # Base dataset, no augmentations
    train_dset_noaug, test_dset = data_fun(imagenet_size=args.timm_models)
    train_dset = []
    for i in range(args.n_pop):
        train_dset += [data_fun(imagenet_size=args.timm_models, 
            re=float(hyperparams[i]['re'])
            )[0]]

    # We extract an held-out portion of the data as validation dataset
    if args.val_perc > 0:
        val_n = int(args.val_perc*len(train_dset[0]))
        for i in range(args.n_pop):
            # we ignore the val here, its just removed
            generator = torch.Generator().manual_seed(666)
            train_dset[i] = torch.utils.data.random_split(train_dset[i], [len(train_dset[i]) - val_n, val_n], generator=generator)[0]
        # we extract the train and validation from the non-augmented data
        generator = torch.Generator().manual_seed(666)
        train_dset_noaug, val_dset = torch.utils.data.random_split(train_dset_noaug, [len(train_dset_noaug) - val_n, val_n], generator=generator)
    else:
        val_dset = None

    return args, train_dset_noaug, train_dset, test_dset, val_dset