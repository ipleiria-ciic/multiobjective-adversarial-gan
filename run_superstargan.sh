#!/bin/sh

EPOCHS=100000
ATTACKS="FGSM TRM UAP SGA"
DELTAS="0.01 0.05 0.10 0.15 0.20"

for ATTACK in $ATTACKS; do
    for DELTA in $DELTAS; do
        python SuperstarGAN/main.py --num_iters $EPOCHS --attack $ATTACK --delta $DELTA --dataset_train Dataset/$ATTACK/$ATTACK-$DELTA
    done
done