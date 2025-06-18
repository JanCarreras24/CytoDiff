# Final Degree Project – AI-Driven Cytomorphology Image Synthesis for Medical Diagnostics


This repository contains the full codebase for the Final Degree Project in Bioinformatics, developed at Helmholtz Munich. The project aims to improve blood cell image classification by augmenting real data with high-quality synthetic images generated using diffusion models and LoRA-based fine-tuning.

## 🔬 Project Pipeline

The structure of this repository follows a three-stage pipeline:

1. **LoRA Training**  
2. **Image Generation**  
3. **Image Classification**

Each step depends on the successful completion of the previous one. It is essential to execute them in the correct order.

---

## ⚙️ Setup

Before starting, make sure to install all required packages listed in the `requirements/` directory. You may use a virtual environment (e.g. `venv`, `conda`) to isolate dependencies.

```bash
pip install -r requirements/requirements.txt
```

---

## 🧪 1. LoRA Training

To fine-tune Stable Diffusion with your real blood cell images, navigate to the `training/` directory.

- Configure the training scripts with correct paths to your **real image dataset**
- Edit the dataset-related parameters to fit your data format
- Train LoRA weights using the provided scripts

The result will be a set of LoRA weights that adapt the diffusion model to your specific cell types.

---

## 🧬 2. Image Generation

Once the LoRA weights are trained, you can generate synthetic images using **Stable Diffusion v2.1**.

- Download the base diffusion model (Stable Diffusion 2.1)
- Load the trained LoRA weights
- Fill in the appropriate paths in the scripts under `generation/`

This step will output synthetic, high-resolution blood cell images that mimic your real data distribution.

---

## 🧠 3. Classification

Finally, use the generated images to train classification models (ResNet-50 or CLIP).

Inside the `classification/` directory:

- Prepare the preprocessing steps required for the dataloader
- Train CNN-based classifiers on real, synthetic, or mixed (real + synthetic) datasets

You can modify the dataloader pipeline to suit your structure if needed.

---

## 📂 Repository Structure

```
Final-Degree-Project/
├── classification/          # Classifier training and evaluation
├── generation/              # Synthetic image generation
├── training/                # LoRA fine-tuning for Stable Diffusion
├── requirements/            # Required Python dependencies
```

---

## 📎 Supplementary Material

- 📄 [Main Manuscript](https://github.com/JanCarreras24/Final-Degree-Project/releases): Full project report
- 📑 [Supplementary Material](https://github.com/JanCarreras24/Final-Degree-Project/releases): Additional tables and figures

---

## 👨‍💻 Author

**Jan Carreras Boada**  
Bachelor's Degree in Bioinformatics  
ESCI-UPF & Helmholtz Munich  
June 2025

---

## 📜 License

This repository is intended for academic and research purposes only.  
If you have any questions or need further information, feel free to contact me at **jancarreras24@gmail.com**.
