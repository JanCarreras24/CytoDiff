import re

# === Input de ejemplo ===
input_text = """
val_top1 = 0.7058 ± 0.03
val_f1_macro = 0.4351 ± 0.03
val_auc_macro = 0.9227 ± 0.00
test_top1 = 0.6190 ± 0.02
test_f1_macro = 0.4269 ± 0.02
test_auc_macro = 0.8863 ± 0.01
"""

# === Extraer valores ===
pattern = r"(\w+)\s*=\s*([\d.]+)\s*±\s*([\d.]+)"
matches = re.findall(pattern, input_text)
metrics = {key: (mean, std) for key, mean, std in matches}

# === Generar tabla LaTeX ===
latex = r"""\begin{table}[ht]
    \centering
    \caption{Classification performance of the fine-tuned CLIP model trained on all real images and 100 additional synthetic images per class. Results are averaged over 5 cross-validation folds.}
    \vspace{-0.3cm}
    \tabcolsep=0.03\linewidth
    \resizebox{0.8\textwidth}{!}{
    \begin{tabular}{lcc}
    \toprule
    \textbf{Metric} & \textbf{Validation} & \textbf{Test} \\
    \midrule
"""

# Añadir filas
rows = [
    ("Accuracy", "val_top1", "test_top1"),
    ("F1 Macro", "val_f1_macro", "test_f1_macro"),
    ("AUC", "val_auc_macro", "test_auc_macro")
]

for label, val_key, test_key in rows:
    val_mean, val_std = metrics[val_key]
    test_mean, test_std = metrics[test_key]
    latex += f"    {label:<18} & {val_mean} \\scriptsize$\\pm{val_std}$ & {test_mean} \\scriptsize$\\pm{test_std}$ \\\\\n"

latex += r"""    \bottomrule
    \end{tabular}}
    \label{tab:clip_100synthetic}
\end{table}
"""

# === Guardar en archivo ===
with open("results_table.tex", "w") as f:
    f.write(latex)

print("✅ Tabla LaTeX guardada como 'results_table.tex'")
