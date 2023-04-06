#### PAPA-gradual experiments, baseline models are in the other paper_experiments.sh file

#### Imagenet

# Note these were ran with a single 32Gb V100 GPU (so for pop=3, its 3 times slower) for minimal queueing time
# For parallel training with maximum speed using 3 gpus, do: "--pop 3 --world_size 3"

# pop=2 epochs=50 no-data-aug
myargs="--data imagenet --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 2 --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh

# pop=2 epochs=50 data-aug-clean
myargs="--data imagenet --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 0.10 --n_pop 2 --hyperparams_mix False --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh


# pop=3 epochs=50 no-data-aug
myargs="--data imagenet --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh

# pop=3 epochs=50 data-aug-clean
myargs="--data imagenet --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 50 --multisteplr_mile .33 .66 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 0.10 --n_pop 3 --hyperparams_mix False --every_k_epochs 1 --method_comb avg --repair True"
python experiments_imagenet.sh


##### Replication of CIFAR-10 in ESGD paper (https://arxiv.org/pdf/1810.06773.pdf)

# avg
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
python experiments.sh


##### CIFAR-100

## cifar-100 Resnet18

myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh

## cifar-100 Resnet18 mixup+smooth

myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh


## cifar-10 Resnet18

myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh

## cifar-10 Resnet18 mixup+smooth

myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 3 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 5 --method_comb avg"
python experiments.sh

myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation True --n_iter_matching 9999 --repair_soup True --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb none"
python experiments.sh
myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler multisteplr --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 10 --every_k_epochs 5 --method_comb avg"
python experiments.sh



