#!/bin/bash

SRC1="/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"
SRC2="/home/aih/jan.boada/project/codes/results/synthetic2/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"

for class_dir in "$SRC2"/*; do
    class_name=$(basename "$class_dir")
    src_class_path="$class_dir"
    dest_class_path="$SRC1/$class_name"

    # Asegurarse que el destino existe
    mkdir -p "$dest_class_path"

    i=500
    for img in "$src_class_path"/*.png; do
        cp "$img" "$dest_class_path/$i.png"
        ((i++))
    done
done
