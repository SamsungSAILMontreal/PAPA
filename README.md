<h1 align="center"> PopulAtion Parameter Averaging (PAPA) </h1>

<a href="https://arxiv.org/abs/2304.03094" target="_blank">Paper</a>, <a href="https://ajolicoeur.wordpress.com/papa" target="_blank">Blog</a> </h3>

This is the official implementation of the paper [PopulAtion Parameter Averaging (PAPA)](https://openreview.net/forum?id=cPDVjsOytS) published at TMLR. In this paper, we devise an algorithm to obtain a strong single model by training a population of models and averaging them once-in-a-while or slowly pushing them toward the average. Through PAPA, one can obtain a better model than by training a single model. See the [blog post](https://ajolicoeur.wordpress.com/papa) for a summary of the approach.

![](https://github.com/AlexiaJM/recombine_nets/blob/clean_for_release/assets/Old_Merging.gif)

## Installation

```
# Assuming python=3.8.2 cuda=11.4 cudnn=8.2.0
pip install -r requirements.txt # install all requirements
```

## Experiments

The experiments to reproduce the paper can be found in /scripts/paper_experiments.sh and /scripts/paper_gradual_experiments.sh.


### How to train

You can train a population of 5 models using PAPA (previously called PAPA-gradual), PAPA-all, PAPA-2, or using no averaging using:
```
python main.py --data cifar100 --model_name resnet18 --batch_size 64 --n_pop 5 --mixup 0.0 0.5 1.0 --smooth 0.00 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --every_k_epochs 5 --ema_alpha 0.99 --ema_every_k 10" # PAPA
python main.py --data cifar100 --model_name resnet18 --batch_size 64 --n_pop 5 --mixup 0.0 0.5 1.0 --smooth 0.00 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --every_k_epochs 5 --method_comb avg" # PAPA-all
python main.py --data cifar100 --model_name resnet18 --batch_size 64 --n_pop 5 --mixup 0.0 0.5 1.0 --smooth 0.00 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --every_k_epochs 5 --method_comb pair_half" # PAPA-2
python main.py --data cifar100 --model_name resnet18 --batch_size 64 --n_pop 5 --mixup 0.0 0.5 1.0 --smooth 0.00 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --every_k_epochs 5 --method_comb none" # no averaging (independently trained models)
```

Output results will be printed and also saved in `final_results.txt`. Note that ```--method_comb none``` apply REPAIR to model soups and 
will automatically use permutation-alignment exclusively for VGG and Resnet18 networks (permutation-alignment is not implemented for other neural architectures).

### Important options

Some of the important options are highlighted below:

```
results_file='/scratch/jolicoea/recombine_nets/final_results.txt' # where to store the results
optim='sgd' # 'sgd' (recommended to use lr=1e-1, lr_min=1e-4), 'adamw' (recommended to use lr=1e-4, lr_min=1e-6)
wd=1e-4 # weight decay for SGD or AdamW
data='cifar100' # 'cifar10', 'cifar100', or 'imagenet'
val=0.02 # percentage of train data held-out to be taken as validation data
model_name='resnet18' # 'vgg11', 'vgg13', 'vgg15', 'resnet18', 'resnet20', 'resnet50' (imagenet-only)
n_pop=5 # size of the population
batch_size=64 # batch-size
every_k_epochs=5 # average every K epochs; for large datasets like imagenet use 1, for small datasets use 5-10, for fine-tuning small datasets use 20
method_comb=avg # 'none' (no-averaging), 'avg' (PAPA-all), 'pair_half' (PAPA-2); other options are also available, see the code
ema_alpha=0.99 # If not equal to 1.0, it will use PAPA (PAPA-gradual); w <- ema_alpha*w + (1-ema_alpha)*w_average (0.99 is recommended when training from scratch, 0.9995 is recommended when fine-tuning a pretrained model)
ema_every_k=10 # apply EMA of PAPA every k SGD steps (10 is recommended)
repair=True # optional REPAIR step
n_iter=5 # Number of forward passes used with REPAIR; for large datasets like imagenet use 25, for small datasets use 5
timm_models=False # If True, uses models from the timm library (so choose model_name accordingly)
finetune=False # If True, load a pretrained model from the timm library for the chosen model
```


## Regularizations

In the paper, we recommend using different data-augmentations/regularizations in each network of the population. In the paper, we do this on Mixup, label smoothing, CutMix, and Random Erasing; note that varying more types of regularizations should be beneficial, but these are the ones we used arbitrarily.

To randomly choose between combinations of regularizations, you can do:
```
--mixup 0.0 0.5 1.0 --smooth 0.00 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35
```

To use no regularization, you can do:
```
--mixup 0.0 --smooth 0.00 --cutmix 0.0 --re 0.0
```


## Parallel or Multi-GPU Training

Currently, the code implements the following options:
1. single GPU training; this will for-loop through the ```n_pop``` networks
2. non-parallel multi-GPU training for when you have a large model or batch size, and you need your multiple GPUs to handle a single network; this will for-loop through the ```n_pop``` networks; this is done automatically when ```--world_size=1``` and you have multiple GPUs
3. parallel multi-GPU single-GPU-per-network (specify the number of GPUs in ```world_size```); for example you have 3 GPUs and 10 networks (```--n_pop 10```) use ```--world_size=3``` so that GPU-0 will train networks 1-2-3, GPU-1 will train networks 4-5-6, and GPU-2 will train networks 7-8-9-10. Thus, instead of a for-loop over 10 networks, each GPU will for-loop over 3 or 4 networks.

To use parallel training (option 3), simply specify the number of GPUs you use in the argument ```world_size```. Note that if you Ctrl-c or suddenly close the Python process, it may still continue as a background process. Make sure to do ```top``` in bash to see if you still have a currently running process of main.py; if so, you can kill it by doing ```kill PID_number```.

Currently, the only option missing is parallel multi-GPU training with multiple-GPUs-per-network. For example, say you use a large mini-batch or architecture, and you need 2 GPUs to be able to do one optimization step on a single network; then you would want to use your 4 GPUs so that GPU-0 and GPU-1 handle the first half of the population and GPU-2, GPU-3 handle the second half of the population. Sadly this option is not implemented. If you are familiar with torch.distributed, feel free to propose a pull request to make it possible. I assume it should be possible using DistributedDataParallel.

Caveat: 2 GPUs in parallel may lead to 1.3-1.5 times faster but do not expect it to be 2 times faster; there is a bottleneck that slows down the parallelization and its gathering and scattering the networks to build the average network (and optionally use REPAIR) with GPU-0 and also to send this average network back to each GPUs. Sadly, the scatter and gather operations in PyTorch are very slow. Feel free to propose a pull request if you know ways of making it more efficient.

Note that PAPA has not been implemented to allow parallel mode; this is because it's not worthwhile to parallelize it, given the frequent and painfully slow scatter/gather operations of PyTorch. Ideally, one should make a asynchronous version of PAPA.

I suspect the best way to obtain fast and painless parallelization would be to rewrite the code in Jax. Jax can easily parallelize a for-loop over multiple GPUs (using pmap), so one could likely parallelize a loop containing both samples from the mini-batch and the multiple models of the population.

## Datasets

### CIFAR-10, CIFAR100

These datasets will be downloaded automatically.

### Imagenet

Download Imagenet from https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2 and https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5 and put both ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar into the same folder as this code. You also need to download https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh and put it in the same folder as the code. Use experiments_imagenet.sh since the data processing will be done there.

### Timm dependency

The latest timm version is needed to do the fine-tuning using the latest neural architectures.

You can install the latest version of timm using:
```
pip install --no-deps git+https://github.com/rwightman/pytorch-image-models
```

If you cannot install it due to a problem with the safetensors library (I have this problem), use the following commands to install it:
```
pip install --upgrade huggingface-hub pyyaml # dependencies
pip install --no-deps git+https://github.com/rwightman/pytorch-image-models # skipping the dependencies because safetensors cannot be installed
```

## References

If you find the code/idea useful for your research, please consider citing:


```bib
@article{jolicoeurmartineau2024population,
title={PopulAtion Parameter Averaging ({PAPA})},
author={Alexia Jolicoeur-Martineau and Emy Gervais and Kilian FATRAS and Yan Zhang and Simon Lacoste-Julien},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=cPDVjsOytS},
note={}
}
```
