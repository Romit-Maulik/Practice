#!/bin/bash
#
#SBATCH --job-name=gpu-test
#SBATCH --account=AIEADA-2
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:8

module load anaconda3
conda activate graphweather

# find the ip-address of one of the node. Treat it as master
ip1=`hostname -I | awk '{print $2}'`
echo $ip1

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

srun python ddp_training.py --nodes 2 --gpus 8 --epochs 100 --ip_address $ip1

# Mashed together from
# https://tuni-itc.github.io/wiki/Technical-Notes/Distributed_dataparallel_pytorch/
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
# https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-distributed.py
