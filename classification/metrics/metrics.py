import os
import re
import json
import csv

def parse_log_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    min_loss = float("inf")
    best_val = {}

    for line in lines:
        if line.startswith("Validation stats for epoch"):
            match = re.search(r"epoch (\d+): ({.*})", line)
            if match:
                epoch = int(match.group(1))
                metrics = json.loads(match.group(2).replace("'", '"'))
                loss = metrics.get("val/loss", float("inf"))
                if loss < min_loss:
                    min_loss = loss
                    best_val = {
                        "epoch": epoch,
                        "val/top1": metrics.get("val/top1"),
                        "val/f1_macro": metrics.get("val/f1_macro"),
                        #"val/f1_micro": metrics.get("val/f1_micro"),
                        "val/auc_macro": metrics.get("val/auc_macro")
                    }

    test_metrics = {}
    test_line_index = next((i for i, l in enumerate(lines) if "Test stats:" in l), None)
    if test_line_index is not None:
        try:
            test_metrics = json.loads(lines[test_line_index].split("Test stats:")[1].strip().replace("'", '"'))
        except:
            pass

    val_top1 = best_val.get("val/top1")
    test_top1 = test_metrics.get("test/top1")

    return {
        #"best_val_epoch": best_val.get("epoch"),  # Eliminada la epoch para no exportarla
        "val_top1": val_top1 / 100 if val_top1 is not None else None,
        "val_f1_macro": best_val.get("val/f1_macro"),
        #"val_f1_micro": best_val.get("val/f1_micro"),
        "val_auc_macro": best_val.get("val/auc_macro"),
        "test_top1": test_top1 / 100 if test_top1 is not None else None,
        "test_f1_macro": test_metrics.get("test/f1_macro"),
        #"test_f1_micro": test_metrics.get("test/f1_micro"),
        "test_auc_macro": test_metrics.get("test/auc_macro")
    }

def collect_all_results(base_dir, experiment_list):
    all_results = []
    for exp in experiment_list:
        exp_path = os.path.join(base_dir, str(exp))
        for fold in range(5):
            log_path = os.path.join(exp_path, f"fold{fold}", "matek", "clipViT-B", "32" ,"log.log") #resnet50
            if os.path.exists(log_path):
                metrics = parse_log_file(log_path)
                metrics["fold"] = fold
                metrics["experiment"] = exp
                all_results.append(metrics)
            else:
                print(f"WARNING: No existe el archivo {log_path}")
    return all_results

def save_results_to_csv(results, output_file):
    if not results:
        print("No hay resultados para guardar.")
        return

    keys = ["experiment", "fold",  # "best_val_epoch" "val_f1_micro" "test_f1_micro" removed
            "val_top1", "val_f1_macro", "val_auc_macro", 
            "test_top1", "test_f1_macro", "test_auc_macro"]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Resultados guardados en {output_file}")

# Parámetros y ejecución:
base_experiment_dir = "/home/aih/jan.boada/project/codes/results/classification/clip/mix"
experiments = list(range(0, 1100, 100))  # [100, 200, ..., 1000]
experiments.append(2000)
experiments.append(3000)
experiments.append("balanced3000")
experiments.append("synthetic")



results = collect_all_results(base_experiment_dir, experiments)
save_results_to_csv(results, "results_5folds_clip.csv")

