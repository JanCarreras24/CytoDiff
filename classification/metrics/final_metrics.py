import pandas as pd

def summarize_results(input_csv, output_csv_summary, output_txt_summary=None):
    # Cargar CSV original
    df = pd.read_csv(input_csv)

    # Columnas métricas (excepto experiment y fold)
    metrics = [col for col in df.columns if col not in ["experiment", "fold"]]

    # Agrupar por experimento y calcular media y std
    summary = df.groupby("experiment")[metrics].agg(['mean', 'std'])

    # Flatten columnas MultiIndex
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index(inplace=True)

    # Guardar resumen a CSV con toda la precisión
    summary.to_csv(output_csv_summary, index=False)
    print(f"Resumen guardado en {output_csv_summary}")

    # Opcional: generar archivo TXT con formato tipo "metric = mean ± std" con 2 decimales
    if output_txt_summary:
        with open(output_txt_summary, 'w') as f:
            for _, row in summary.iterrows():
                exp = row["experiment"]
                f.write(f"--- Experimento {exp} ---\n")
                for metric in metrics:
                    mean_val = row[f"{metric}_mean"]
                    std_val = row[f"{metric}_std"]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # media con 4 decimales, sd con 2 decimales
                        f.write(f"{metric} = {mean_val:.4f} ± {std_val:.2f}\n")
                f.write("\n")
        print(f"Resumen en texto guardado en {output_txt_summary}")


# Uso:
input_csv = "results_5folds_clip.csv"
output_csv_summary = "final_metrics_clip.csv"
output_txt_summary = "final_metrics_clip.txt"  # opcional

summarize_results(input_csv, output_csv_summary, output_txt_summary)
