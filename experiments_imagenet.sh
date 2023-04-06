#!/bin/bash

## sync files
#rsync --exclude='.git/' -avz --no-g --no-p /scratch/jolicoea/recombine_nets $SLURM_TMPDIR/ ### REMOVE
#cd $SLURM_TMPDIR/recombine_nets ### REMOVE

## load modules
#module load python/3.8.2 StdEnv/2020 gcc/9.3.0 cuda/11.4 cudacore/.11.4.2 cudnn/8.2.0 scipy-stack/2020b ### REMOVE
#source /scratch/jolicoea/gitrebasin/env/bin/activate ### REMOVE

## get imagenet
# wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
#rsync --exclude='.git/' -avz --no-g --no-p /scratch/jolicoea/torchvision_train/ILSVRC2012_img_train.tar $SLURM_TMPDIR/recombine_nets/ILSVRC2012_img_train.tar ### REMOVE
#rsync --exclude='val/' -avz --no-g --no-p /scratch/jolicoea/torchvision_val/ILSVRC2012_img_val.tar $SLURM_TMPDIR/recombine_nets/ILSVRC2012_img_val.tar ### REMOVE
#rsync --exclude='.git/' -avz --no-g --no-p /scratch/jolicoea/valprep.sh $SLURM_TMPDIR/recombine_nets/valprep.sh ### REMOVE

# Instructions from https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
bash ../valprep.sh
cd ..

python main.py ${myargs}
