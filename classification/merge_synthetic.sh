#!/bin/bash

SRC1="/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"
SRC2="/home/aih/jan.boada/project/codes/results/synthetic2/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"

# Para cada carpeta de clase en synthetic2
for class_dir in "$SRC2"/*; do
    class_name=$(basename "$class_dir")
    src_class_path="$class_dir"
    dest_class_path="$SRC1/$class_name"

    # Crear directorio destino si no existe
    mkdir -p "$dest_class_path"

    # Contar cuántos archivos hay en el destino para esa clase (asumiendo que ya hay 3000 en synthetic)
    # Esto es para saber a partir de qué número nombrar las imágenes nuevas
    count_existing=$(ls "$dest_class_path"/*.png 2>/dev/null | wc -l)
    
    # Si no hay imágenes, poner count_existing a 0
    if [ "$count_existing" == "0" ]; then
        count_existing=0
    fi

    i=$count_existing

    # Copiar imágenes de synthetic2, renombrándolas empezando desde el número count_existing
    for img in "$src_class_path"/*.png; do
        cp "$img" "$dest_class_path/$i.png"
        ((i++))
    done

done
