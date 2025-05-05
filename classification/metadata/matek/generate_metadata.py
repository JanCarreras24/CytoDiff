import os
import random
from sklearn.model_selection import train_test_split

# Paths (cambia según tus rutas)
real_train_fewshot_dir = "/ictstr01/home/aih/jan.boada/project/codes/datasets/data/matek/real_train_fewshot/seed0"
output_train_dir = "/ictstr01/home/aih/jan.boada/project/codes/classification/metadata/matek/train"
output_test_dir = "/ictstr01/home/aih/jan.boada/project/codes/classification/metadata/matek/test"

# Mapea las clases a un número (según tu ejemplo, ajusta a tus clases)
label_map = {
    'Basophil': 0, 'Eosinophil': 1, 'Erythroblast': 2, 'Atypical Lymphocyte': 3, 
    'Typical Lymphocyte': 4, 'Metamyelocyte': 5, 'Monoblast': 6, 'Monocyte': 7, 
    'Myeloblast': 8, 'Myelocyte': 9, 'Band Neutrophil': 10, 'Segmented Neutrophil': 11, 
    'Promyelocyte': 12, 'Promyelocyte Bilobed': 13, 'Smudge cell': 14
}

# Crear las carpetas de salida si no existen
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Inicializar los archivos que se van a escribir
train_class_labels = open(os.path.join(output_train_dir, "class_labels.txt"), "w")
train_image_ids = open(os.path.join(output_train_dir, "image_ids.txt"), "w")
test_class_labels = open(os.path.join(output_test_dir, "class_labels.txt"), "w")
test_image_ids = open(os.path.join(output_test_dir, "image_ids.txt"), "w")

# Recorrer las clases y dividir por cada clase
for class_name, label in label_map.items():
    class_folder = os.path.join(real_train_fewshot_dir, class_name)
    
    # Asegurarse de que la carpeta existe
    if not os.path.exists(class_folder):
        print(f"La carpeta {class_folder} no existe.")
        continue
    
    # Listar las imágenes de la clase
    images = [f for f in os.listdir(class_folder) if f.endswith('.tiff')]  # Asumiendo que las imágenes son .tiff
    
    # Dividir en train y test
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Escribir los archivos para Train
    for img in train_imgs:
        image_path = os.path.join(class_name, img)
        train_class_labels.write(f"{image_path},{label}\n")
        train_image_ids.write(f"{image_path}\n")
    
    # Escribir los archivos para Test
    for img in test_imgs:
        image_path = os.path.join(class_name, img)
        test_class_labels.write(f"{image_path},{label}\n")
        test_image_ids.write(f"{image_path}\n")

# Cerrar los archivos después de escribir
train_class_labels.close()
train_image_ids.close()
test_class_labels.close()
test_image_ids.close()

print("Partición de datos completada y archivos generados.")
