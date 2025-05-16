#!/bin/sh

G_CHECKPOINT=100000
E_CHECKPOINT=200
ATTACKS="SGA TRM FGSM UAP"
DELTAS="0.01 0.05 0.10 0.15 0.20"

for ATTACK in $ATTACKS; do
    for DELTA in $DELTAS; do
        python Testing/testing.py --attack $ATTACK --delta $DELTA --checkpoint_gan $G_CHECKPOINT --checkpoint_encoder $E_CHECKPOINT 
    done
done