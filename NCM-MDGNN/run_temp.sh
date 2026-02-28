#!/bin/bash
MESHFILES=/xuzhewen/gnn/data_process/meshcnn

if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi


# create log directory
mkdir -p logs

python train.py \
--batch-size 2 \
--forecast_len 1 \
--test-batch-size 2 \
--epochs 50 \
--data_folder data/data \
--max_level 5 \
--min_level 0 \
--feat 8 \
--log_dir log_f8_balance \
--decay \
--lr 1e-2 \
--balance
