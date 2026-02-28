#!/bin/bash
MESHFILES=/home/xzw/meshcnn
batch_size=16
if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# create log directory
mkdir -p logs

# torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="127.0.0.1" --master_port=2333 train.py \

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -W ignore rollout.py \
--batch-size $batch_size \
--test-batch-size $batch_size \
--epochs 1 \
--data_folder /data2/xzwdir/2011-2021 \
--climatology_dir /data2/xzwdir/climtology/ \
--max_level 6 \
--min_level 0 \
--feat 32 \
--log_dir log_f8_balance \
--decay \
--lr 1e-4 \
--balance 