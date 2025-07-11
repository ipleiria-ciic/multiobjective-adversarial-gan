#!/bin/sh

BLURS="gaussian median bilateral affine"

for BLUR in $BLURS; do
    python Defences/defences_blur.py --blur $BLUR
done