
#### Note: Its inevitable that you will probably not get the same exact numbers as in the paper.
# By heavily cleaning the code, I removed a lot of unecessary garbage so the randomness is likely changed.

#### Imagenet

# Note these were ran with a single 32Gb V100 GPU (so for pop=3, its 3 times slower) for minimal queueing time
# For parallel training with maximum speed using 3 gpus, do: "--pop 3 --world_size 3"

# pop=3 epochs=50 no-data-aug
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 5 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 1 --method_comb none"
python experiments_imagenet.sh
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh

# pop=3 epochs=50
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 1 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.2 --wd 1e-4 --smooth 0.0 0.10 --n_pop 3 --hyperparams_mix False --every_k_epochs 1 --method_comb none"
python experiments_imagenet.sh
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.2 --wd 1e-4 --smooth 0.0 0.10 --n_pop 3 --hyperparams_mix False --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh

# pop=2 epochs=50 no-data-aug
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 5 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 2 --every_k_epochs 1 --method_comb none"
python experiments_imagenet.sh
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 2 --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh

# pop=2 epochs=50
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 1 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 0.10 --n_pop 2 --hyperparams_mix False --every_k_epochs 1 --method_comb none"
python experiments_imagenet.sh
myargs="--data imagenet --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 0.10 --n_pop 2 --hyperparams_mix False --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh


##### Replication of CIFAR-10 in ESGD paper (https://arxiv.org/pdf/1810.06773.pdf)

myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 1 --every_k_epochs 5 --method_comb none --mixed_precision False --grad_scaler False" 
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none --mixed_precision False --grad_scaler False" 
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none --mixed_precision False --grad_scaler False" 
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none --mixed_precision False --grad_scaler False" 
python experiments.sh

# avg
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
python experiments.sh

# pair_half
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb pair_half --mixed_precision False --grad_scaler False"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --mixed_precision False --grad_scaler False"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb pair_half --mixed_precision False --grad_scaler False"
python experiments.sh


##### CIFAR-100

## cifar-100 VGG-16

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none " 
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

## cifar-100 VGG16 mixup+smooth

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh



## cifar-100 Resnet18

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

## cifar-100 Resnet18 mixup+smooth

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

# test permutations and correlation-est

myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none --n_iter_matching 9999 --repair_soup True --correlation_est True"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --correlation_est True"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --correlation_est True"
python experiments.sh


######################


## cifar-10 VGG-11

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

## cifar-10 VGG11 mixup+smooth

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

## cifar-10 Resnet18

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

## cifar-10 Resnet18 mixup+smooth

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh
myargs=" --data cifar10 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb pair_half"
python experiments.sh



### Ablation cifar-100 resnet

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb greedy_soup --same_init False  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb many_half --same_init False  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_75 --same_init False  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb many_75 --same_init False  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none --same_init False  --repair True  --n_iter 5"
python experiments.sh

# avg stuff

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --repair True  --n_iter 5 --mutation_sigma 0.01"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False   --repair False  --n_iter 5"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 1 --method_comb avg --same_init False   --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 10 --method_comb avg --same_init False   --repair True  --n_iter 5"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init True   --repair True  --n_iter 5"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation True --weight_matching False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 10 --method_comb avg    --repair True  --n_iter 5"
python experiments.sh


# pair-half stuff

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --repair True  --n_iter 5 --mutation_sigma 0.01"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False   --repair False  --n_iter 5"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 1 --method_comb pair_half --same_init False   --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 10 --method_comb pair_half --same_init False   --repair True  --n_iter 5"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init True   --repair True  --n_iter 5"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300 --permutation True --weight_matching False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 10 --method_comb pair_half    --repair True  --n_iter 5"
python experiments.sh



### With tournament selection
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --tournament_pairwise True  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --mutation_sigma 0.01 --tournament_pairwise True  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --selection True --tournament_pairwise True  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --selection True --elitism_maxiter 1 --tournament_pairwise True  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --selection True --elitism_maxiter 1 --mutation_sigma 0.01 --tournament_pairwise True  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --selection True --mutation_sigma 0.01 --tournament_pairwise True  --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False --selection True --mutation_sigma 0.01 --tournament_pairwise True  --repair False  --n_iter 5"
python experiments.sh

# varying amount of merging avg
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0 0.25 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0 0.5 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0 0.75 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0.75 1 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0.5 1 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0.25 1 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --same_init False  --range_merge 0.25 0.75 --repair True  --n_iter 5"
python experiments.sh

# varying amount of merging pair_half
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0 0.25 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0 0.5 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0 0.75 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0.75 1 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0.5 1 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0.25 1 --repair True  --n_iter 5"
python experiments.sh
myargs=" --data cifar100 --EPOCHS 300 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb pair_half --same_init False  --range_merge 0.25 0.75 --repair True  --n_iter 5"
python experiments.sh


#### Showing no benefit of training one long run over using PAPA n-pop=5

myargs=" --data cifar10 --EPOCHS 300*5 --n_iter 0 --permutation False --n_iter_matching 9999 --repair_soup False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 1 --every_k_epochs 9999 --method_comb none"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300*5 --n_iter 0 --permutation False --n_iter_matching 9999 --repair_soup False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 1 --every_k_epochs 9999 --method_comb none"
python experiments.sh

myargs=" --data cifar10 --EPOCHS 300*5 --n_iter 0 --permutation False --n_iter_matching 9999 --repair_soup False --model_name VGG11 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 1 --every_k_epochs 9999 --method_comb none"
python experiments.sh

myargs=" --data cifar100 --EPOCHS 300*5 --n_iter 0 --permutation False --n_iter_matching 9999 --repair_soup False --model_name VGG16 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 1 --every_k_epochs 9999 --method_comb none"
python experiments.sh


