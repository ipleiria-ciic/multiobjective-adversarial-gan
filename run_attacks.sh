#!/bin/sh

ATTACKS="SGA TRM FGSM UAP"
DELTAS="0.01 0.05 0.10 0.15 0.20"

for ATTACK in $ATTACKS; do
    for DELTA in $DELTAS; do
        python Attacks/$ATTACK.py --delta $DELTA
    done
done