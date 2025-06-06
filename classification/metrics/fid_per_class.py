import os
import random
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd

# --- Configuración ---
real_dir = "/home/aih/jan.boada/project/codes/datasets/jan/data/matek/real_train_fewshot/seed0"
#"/ictstr01/home/aih/jan.boada/project/codes/datasets/data/matek/real_train_fewshot/seed0"
synthetic_dir = "/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"
output_csv = "fid_per_class16_jan3.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64

# Nuevo parámetro: usar tamaño fijo por clase
use_fixed_sample_size = True
fixed_sample_size = 16  # solo se usa si use_fixed_sample_size = True

transform = transforms.Resize((299, 299))

# --- Funciones ---
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        tensor = pil_to_tensor(img)
        return tensor
    except Exception as e:
        print(f"[ERROR] Fallo cargando imagen {path}: {e}")
        return None

def load_images(folder, extensions):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(extensions)]
    return [img for img in map(load_image, files) if img is not None]

def update_fid_in_batches(fid, tensors, real: bool):
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i+batch_size]
        batch_tensor = torch.stack(batch).to(device)
        fid.update(batch_tensor, real=real)
        del batch_tensor
        torch.cuda.empty_cache()

# --- Loop principal ---
fid_records = []
classes = sorted(os.listdir(real_dir))

for cls in tqdm(classes, desc="Calculando FID por clase"):
    real_path = os.path.join(real_dir, cls)
    synth_path = os.path.join(synthetic_dir, cls)

    if not os.path.exists(real_path) or not os.path.exists(synth_path):
        print(f"[{cls}] Ruta no encontrada, se omite")
        continue

    real_imgs = load_images(real_path, ('.tiff',))
    synth_imgs = load_images(synth_path, ('.png',))

    if use_fixed_sample_size:
        sample_size = fixed_sample_size
    else:
        sample_size = min(len(real_imgs), len(synth_imgs))

    if len(real_imgs) < sample_size or len(synth_imgs) < sample_size:
        print(f"[{cls}] No hay suficientes imágenes para usar {sample_size} (real: {len(real_imgs)}, synth: {len(synth_imgs)}), se omite")
        continue

    real_imgs = random.sample(real_imgs, sample_size)
    synth_imgs = random.sample(synth_imgs, sample_size)

    try:
        fid = FrechetInceptionDistance(feature=2048).to(device)
        update_fid_in_batches(fid, real_imgs, real=True)
        update_fid_in_batches(fid, synth_imgs, real=False)
        score = fid.compute().item()
        fid_records.append({"Clase": cls, "FID": score, "n_samples": sample_size})
    except Exception as e:
        print(f"[{cls}] Error al calcular FID: {e}")
        continue
    finally:
        torch.cuda.empty_cache()

# --- Mostrar y guardar resultados ---
print("\n=== FID por clase (menor = mejor) ===")
for record in sorted(fid_records, key=lambda x: x["FID"]):
    print(f"{record['Clase']}: {record['FID']:.2f} (n={record['n_samples']})")

df = pd.DataFrame(fid_records).sort_values("FID")
df.to_csv(output_csv, index=False)
print(f"\n Resultados guardados en: {output_csv}")
