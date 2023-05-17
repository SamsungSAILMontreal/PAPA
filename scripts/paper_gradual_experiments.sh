#### PAPA-gradual experiments, baseline models are in the other paper_experiments.sh file

########################################################################################

#### Imagenet

# Note these were ran with a single 32Gb V100 GPU (so for pop=3, its 3 times slower) for minimal queueing time
# For parallel training with maximum speed using 3 gpus, do: "--pop 3 --world_size 3"

# pop=3 epochs=50 data-aug
export myargs="--data imagenet --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 90 --n_iter 25 --permutation False --model_name resnet50 --batch_size 256 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.1 0.2 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 3 --every_k_epochs 1 --method_comb avg --repair False"
python experiments_imagenet.sh

########################################################################################


##### Replication of CIFAR-10 in ESGD paper (https://arxiv.org/pdf/1810.06773.pdf)

# avg
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler cosine --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 3 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
bash experiments.sh
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler cosine --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 5 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
bash experiments.sh
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 160 --n_iter 5 --permutation False --model_name resnet20 --batch_size 128 --optim sgd --clip_grad 0 --lr_scheduler cosine --multisteplr_mile .50625 .7625 --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 10 --every_k_epochs 5 --method_comb avg --mixed_precision False --grad_scaler False"
bash experiments.sh


########################################################################################


##### CIFAR-100

## cifar-100 Resnet18

export myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
bash experiments.sh

## cifar-100 Resnet18 mixup+smooth

export myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 3 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 5 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar100 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 10 --every_k_epochs 5 --method_comb avg"
bash experiments.sh


## cifar-10 Resnet18

export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 3 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 5 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 --wd 1e-4 --smooth 0.0 --n_pop 10 --every_k_epochs 5 --method_comb avg"
bash experiments.sh

## cifar-10 Resnet18 mixup+smooth

export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 3 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 5 --every_k_epochs 5 --method_comb avg"
bash experiments.sh
export myargs=" --data cifar10 --ema_every_k 10 --ema_alpha 0.99 --EPOCHS 300 --n_iter 5 --permutation False --model_name Resnet18 --batch_size 64 --optim sgd --clip_grad 0 --lr_scheduler cosine --lr 0.1 --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --cutmix 0.0 0.5 1.0 --re 0.0 0.15 0.35 --n_pop 10 --every_k_epochs 5 --method_comb avg"
bash experiments.sh


########################################################################################


##### Fine-tuning over 50 epochs

# efficientnet n_pop=2
export myargs="--linear_prob_time 0.04 --ema_every_k 10 --ema_alpha 0.9995 --optim adamw --lr_scheduler cosine --lr 1e-4 --lr_min 1e-6 --cutmix 0.00 0.50 1.00 --re 0.00 0.15 0.35 --data cifar100 --EPOCHS 50 --repair_soup False --model_name hf-hub:timm/tf_efficientnetv2_s.in21k --timm_models True --finetune True --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 2 --every_k_epochs 99999 --method_comb avg"
bash experiments.sh

# eva n_pop=4
export myargs="--linear_prob_time 0.04 --ema_every_k 10 --ema_alpha 0.9995 --optim adamw --lr_scheduler cosine --lr 1e-4 --lr_min 1e-6 --cutmix 0.00 0.50 1.00 --re 0.00 0.15 0.35 --data cifar100 --EPOCHS 50 --repair_soup False --model_name hf-hub:timm/eva02_tiny_patch14_224.mim_in22k --timm_models True --finetune True --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 4 --every_k_epochs 99999 --method_comb avg"
bash experiments.sh

# convit n_pop=5
export myargs="--linear_prob_time 0.04 --ema_every_k 10 --ema_alpha 0.9995 --optim adamw --lr_scheduler cosine --lr 1e-4 --lr_min 1e-6 --cutmix 0.00 0.50 1.00 --re 0.00 0.15 0.35 --data cifar100 --EPOCHS 50 --repair_soup False --model_name hf-hub:timm/convit_tiny.fb_in1k --timm_models True --finetune True --mixup 0.0 0.5 1.0 --wd 1e-4 --smooth 0.0 0.05 0.10 --n_pop 5 --every_k_epochs 99999 --method_comb avg"
bash experiments.sh

