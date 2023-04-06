import torchvision
import torchvision.transforms as T
import numpy as np
import PIL

def get_cifar10(imagenet_size=False):
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    normalize = T.Normalize(np.array(CIFAR_MEAN)/255, np.array(CIFAR_STD)/255)
    denormalize = T.Normalize(-np.array(CIFAR_MEAN)/np.array(CIFAR_STD), 255/np.array(CIFAR_STD))
    
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

    train_dset = torchvision.datasets.CIFAR10(root='cifar10', train=True,
                                            download=True, transform=train_transform)
    test_dset = torchvision.datasets.CIFAR10(root='cifar10', train=False,
                                            download=True, transform=test_transform)
    return train_dset, test_dset