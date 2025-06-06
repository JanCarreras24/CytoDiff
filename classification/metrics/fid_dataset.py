import os
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd

# --- Configuración ---
real_dir = "/home/aih/jan.boada/project/codes/datasets/jan/data/matek/real_train_fewshot/seed0"
synthetic_dir = "/home/aih/jan.boada/project/codes/results/synthetic/matek/sd2.1/gs2.0_nis50/shot16_seed6_template1_lr0.0001_ep300/train"
output_csv = "fid_dataset.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64

transform = transforms.Resize((299, 299))

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        tensor = pil_to_tensor(img)
        return tensor
    except Exception as e:
        print(f"[ERROR] Fallo cargando imagen {path}: {e}")
        return None

def load_all_images(root_dir, extensions):
    all_images = []
    for cls in os.listdir(root_dir):
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(extensions):
                    img_path = os.path.join(cls_path, fname)
                    img = load_image(img_path)
                    if img is not None:
                        all_images.append(img)
    return all_images

def update_fid_in_batches(fid, tensors, real: bool):
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i+batch_size]
        batch_tensor = torch.stack(batch).to(device)
        fid.update(batch_tensor, real=real)
        del batch_tensor
        torch.cuda.empty_cache()

# --- Cargar imágenes ---
print("Cargando imágenes reales...")
real_imgs = load_all_images(real_dir, ('.tiff',))
print(f"Total imágenes reales: {len(real_imgs)}")

print("Cargando imágenes sintéticas...")
synth_imgs = load_all_images(synthetic_dir, ('.png',))
print(f"Total imágenes sintéticas: {len(synth_imgs)}")

# --- Recortar al mismo número si son desiguales ---
sample_size = min(len(real_imgs), len(synth_imgs))
real_imgs = real_imgs[:sample_size]
synth_imgs = synth_imgs[:sample_size]

# --- Calcular FID ---
print("Calculando FID a nivel de dataset completo...")
fid = FrechetInceptionDistance(feature=2048).to(device)
update_fid_in_batches(fid, real_imgs, real=True)
update_fid_in_batches(fid, synth_imgs, real=False)
score = fid.compute().item()
print(f"✅ FID para todo el dataset: {score:.2f}")

# --- Guardar en CSV ---
df = pd.DataFrame([{"FID_dataset": score, "n_samples_per_set": sample_size}])
df.to_csv(output_csv, index=False)
print(f"📄 Resultado guardado en: {output_csv}")
