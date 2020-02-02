#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J SearchClient
### -- ask for number of cores (default: 1) -- 
#BSUB -n 12
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=shared:j_exclusive=yes"
#BSUB -R "select[gpu32gb]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=6GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 10GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s144852@student.dtu.dk
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo nn_training.out 
#BSUB -eo nn_training.out


source /dtu/sw/dcc/dcc-sw.bash
nvidia-smi
# Load the cuda module
module load cuda/9.2
/appl/cuda/9.2/samples/bin/x86_64/linux/release/deviceQuery

module load python/3.7.3
pip3 install --user torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# here follow the commands you want to execute 
cd ~/Documents/02285_server
./job.sh
