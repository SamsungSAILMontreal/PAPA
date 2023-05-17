import torchvision
import torchvision.transforms as T
import numpy as np
import PIL
from utils.random_erasing import RandomErasing

def get_imagenet(imagenet_size=False, re=0.0):
    CIFAR_MEAN = [0.485, 0.456, 0.406]
    CIFAR_STD = [0.229, 0.224, 0.225]
    normalize = T.Normalize(np.array(CIFAR_MEAN), np.array(CIFAR_STD))
    
    train_transform = []
    train_transform += [T.RandomResizedCrop(224)] # , interpolation=PIL.Image.BICUBICs
    train_transform += [T.RandomHorizontalFlip()]
    train_transform += [T.ToTensor()]
    train_transform += [normalize]
    if re > 0.0:
        train_transform.append(RandomErasing(re, mode='const', max_count=1, num_splits=0, device='cpu'))
    train_transform = T.Compose(train_transform)

    test_transform = []
    test_transform += [T.Resize(256)] # , interpolation=PIL.Image.BICUBIC
    test_transform += [T.CenterCrop(224)]
    test_transform += [T.ToTensor()]
    test_transform += [normalize]
    test_transform = T.Compose(test_transform)

    train_dset = torchvision.datasets.ImageFolder('train', transform=train_transform)
    test_dset = torchvision.datasets.ImageFolder('val', transform=test_transform)

    return train_dset, test_dset