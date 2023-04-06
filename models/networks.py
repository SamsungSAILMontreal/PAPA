import torchvision
from models.resnet import ResNet18
from models.resnet20 import resnet20
from models.vgg import VGG
import torch
from sync_batchnorm import convert_model
import copy


# Placeholder for Dataparallel so that even when not using it, we still need to do net.module to access the network
class PlaceholderNN(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

def network(args):
    model_name = args.model_name
    num_classes=args.num_classes
    if "vgg" in model_name.lower(): 
        model = VGG(model_name, num_classes=num_classes)
    elif "resnet50" in model_name.lower():  # imagenet
        model = torchvision.models.resnet50(norm_layer=None)
    elif "resnet18" in model_name.lower(): 
        model = ResNet18(model_name, num_classes=num_classes)
    elif "resnet20" in model_name.lower():
        model = resnet20(num_classes=num_classes)
    else:
        raise NotImplementedError("model_name does not exist")
    if args.world_size == 1:
        model = torch.nn.DataParallel(model.to(args.device))
        if args.sync: # SyncBN
            model = convert_model(model)
            model = model.to(args.device)
    else:
        model = PlaceholderNN(model.to(args.device))
    return model

# deepcopy breaks when using distribution so using this function is prefered
def copy_networks(args, nets):
    if hasattr(nets, '__len__'):
        net_list = []
        for i in range(len(nets)):
            net_list += [network(args).to(args.device)]
            net_list[i].load_state_dict(copy.deepcopy(nets[i].state_dict()))
    else:
        net_list = network(args).to(args.device)
        net_list.load_state_dict(copy.deepcopy(nets.state_dict()))
    return net_list
