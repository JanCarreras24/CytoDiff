# CytoDiff – AI-Driven Cytomorphology Image Synthesis for Medical Diagnostics

This repository contains the full codebase developed at Helmholtz Munich. The project aims to improve blood cell image classification by augmenting real data with high-quality synthetic images generated using diffusion models and LoRA-based fine-tuning.

---

## Pipeline Overview

The repository follows a three-stage pipeline:

1. **LoRA Training**  
2. **Image Generation**  
3. **Image Classification**

Each step depends on the successful completion of the previous one and must be executed in order.

---

## Setup

Install all required packages listed in the `requirements/` directory. Using a virtual environment (e.g., `venv` or `conda`) is recommended.

```bash
pip install -r requirements/requirements.txt


---

## 📂 Repository Structure

```
---
CytoDiff/
├── classification/          # Classifier training and evaluation
├── generation/              # Synthetic image generation
├── training/                # LoRA fine-tuning for Stable Diffusion
├── requirements/            # Python dependencies

```

## 📎 Supplementary Material

- 📄 [Main Manuscript](https://github.com/JanCarreras24/Final-Degree-Project/releases): Full project report
- 📑 [Supplementary Material](https://github.com/JanCarreras24/Final-Degree-Project/releases): Additional tables and figures

---


## 📜 License

This repository is intended for academic and research purposes only.  
If you have any questions or need further information, feel free to contact me at **jancarreras24@gmail.com**.


## Code Acknowledgement
This project uses code and resources from the DataDream repository. We thank the authors for making their work publicly available.

