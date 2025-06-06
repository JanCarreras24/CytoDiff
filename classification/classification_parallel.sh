#!/bin/bash

for NUM_IMAGES in 100 200 300 400 500 600 700 800 900 1000 2000 3000 balanced3000
do
    for FOLD in 0 1 2 3 4
    do
        echo "Enviando job: NUM_IMAGES=$NUM_IMAGES, FOLD=$FOLD"
        sbatch classification_parallel.sbatch $NUM_IMAGES $FOLD
    done
done
