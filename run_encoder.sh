#!/bin/sh

EPOCHS=200
G_EPOCH=100000
ATTACKS="SGA TRM FGSM UAP"
DELTAS="0.01 0.05 0.10 0.15 0.20"

for ATTACK in $ATTACKS; do
    for DELTA in $DELTAS; do
        python SuperstarGAN/encoder.py --attack $ATTACK --delta $DELTA --checkpoint_epochs $G_EPOCH
    done
done