import torchvision
import torchvision.transforms as T
import numpy as np
import PIL

def get_cifar100(imagenet_size=False):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]
    normalize = T.Normalize(np.array(CIFAR_MEAN), np.array(CIFAR_STD))
    
    train_transform = []
    train_transform += [T.RandomHorizontalFlip()]
    train_transform += [T.RandomCrop(32, padding=4)]
    if imagenet_size:
        train_transform += [T.Resize(224, PIL.Image.BICUBIC)] # using timm imagenet models so need to rescale to 224
        # can try interpolation=InterpolationMode.BICUBIC, antialias=True also
    train_transform += [T.ToTensor()]
    train_transform += [normalize]
    train_transform = T.Compose(train_transform)

    test_transform = []
    if imagenet_size:
        test_transform += [T.Resize(224, PIL.Image.BICUBIC)] # using timm imagenet models so need to rescale to 224
    test_transform += [T.ToTensor()]
    test_transform += [normalize]
    test_transform = T.Compose(test_transform)

    train_dset = torchvision.datasets.CIFAR100(root='cifar100', train=True,
                                            download=True, transform=train_transform)
    test_dset = torchvision.datasets.CIFAR100(root='cifar100', train=False,
                                            download=True, transform=test_transform)
    return train_dset, test_dset