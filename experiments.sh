#!/bin/bash

## sync files
#rsync --exclude='.git/' -avz --no-g --no-p /scratch/jolicoea/recombine_nets $SLURM_TMPDIR/ ### REMOVE
#cd $SLURM_TMPDIR/recombine_nets ### REMOVE

## load modules
#module load python/3.8.2 StdEnv/2020 gcc/9.3.0 cuda/11.4 cudacore/.11.4.2 cudnn/8.2.0 scipy-stack/2020b ### REMOVE
#source /scratch/jolicoea/gitrebasin/env/bin/activate ### REMOVE

python main.py ${myargs}