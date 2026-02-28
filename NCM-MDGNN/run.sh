#!/bin/bash
MESHFILES=/WdHeDisk/users/xuzhewen/data_process/meshcnn
batch_size=64
if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# create log directory
mkdir -p logs

# torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="127.0.0.1" --master_port=2333 train.py \

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -W ignore train.py \
--batch-size $batch_size \
--test-batch-size $batch_size \
--epochs 1000 \
--max_level 6 \
--min_level 0 \
--seed 64 \
--feat 16 \
--log_dir log_f8_balance \
--decay \
--lr 1e-4 \
--balance
