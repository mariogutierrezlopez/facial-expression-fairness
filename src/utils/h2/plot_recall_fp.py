#   plot_recall_fp.py
#   Mario Gutiérrez López

# Este script grafica las clases donde se está equivocando el modelo en el sesgo representacional

import wandb
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Variables proyecto wandb
WANDB_PROJECT = "mariogutierrezlopez-upm/MultiPIE_test"
CLASS_NAMES = ["Neutral", "Smile", "Surprise", "Squint", "Disgust", "Scream"]

# Estilos gráficas
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
COLOR_PALETTE = sns.color_palette("husl", len(CLASS_NAMES))
COLOR_DICT = {i: COLOR_PALETTE[i] for i in range(len(CLASS_NAMES))}

def get_confusion_matrix_from_wandb(run):
    """Busca y lee la tabla de la matriz de confusión directamente de los archivos de W&B."""
    
    for file in run.files():
        if "test_confusion_matrix" in file.name and file.name.endswith(".json"):
            
            # Descargamos el JSON temporalmente
            file.download(replace=True, root="./temp_logs")
            file_path = os.path.join("./temp_logs", file.name)
            
            with open(file_path, "r", encoding="utf-8") as f:
                table_data = json.load(f)
            
            cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
            
            if "columns" in table_data and "Actual" in table_data["columns"]:
                for row in table_data["data"]:
                    actual, predicted, count = row[0], row[1], row[2]
                    
                    # Convertir el nombre de la clase (String) al índice numérico
                    if isinstance(actual, str):
                        actual = CLASS_NAMES.index(actual)
                    if isinstance(predicted, str):
                        predicted = CLASS_NAMES.index(predicted)
                        
                    cm[actual, predicted] = count
                return cm
            
                
    return None



def main():
    print("Conectando a wandb")
    api = wandb.Api()

    runs = api.runs(WANDB_PROJECT)

    data_by_class = {}

    # Regex para parsear el nombre, tienen formato Stereotipical_bias_c5_f1.0
    name_regex = re.compile(r'Stereotipical_bias_c(\d+)_f([\d\.]+)')

    # OBTENER DATOS DE CADA RUN
    for run in runs:
        match = name_regex.search(run.name)
        if match:
            c_idx = int(match.group(1))
            f_val = float(match.group(2))
            
            print(f"Procesando run {run.name}")
            
            try:
                # Intento 1: Leer el JSON limpio nativo de W&B (El método nuevo)
                cm = get_confusion_matrix_from_wandb(run)

                # Guardado final de la matriz
                if cm is not None:
                    print(f"\n--- Matriz extraída para {run.name} ---")
                    print(cm)
                    print(f"Total muestras en esta matriz: {np.sum(cm)}")
                    
                    if c_idx not in data_by_class:
                        data_by_class[c_idx] = []
                    data_by_class[c_idx].append((f_val, cm))
                else:
                    print(f"⚠️ No se encontró matriz (ni JSON ni log) en {run.name}")
                    
            except Exception as e:
                print(f"Error descargando datos de {run.name}: {e}")
    

    # GENERACION DE GRÁFICAS
    print("Generando gráficas")

    for c_idx, runs_data in data_by_class.items():
        # Ordenar por factor f de menor a mayor
        runs_data.sort(key=lambda x: x[0])
        
        f_values = [rd[0] for rd in runs_data]
        matrices = [rd[1] for rd in runs_data]
        
        num_classes = matrices[0].shape[0]
        
        # Preparar matriz para almacenar tasas de error (Filas: factor f, Columnas: clase predicha)
        error_rates = np.zeros((len(f_values), num_classes))
        
        for i, cm in enumerate(matrices):
            # Coger la fila correspondiente a la clase objetivo (la que tiene el sesgo)
            target_row = cm[c_idx, :]
            total_samples = np.sum(target_row)
            
            if total_samples > 0:
                # Calcular la proporción de predicciones para cada clase
                error_rates[i, :] = target_row / total_samples
                


        plt.figure(figsize=(10, 6))
        target_name = CLASS_NAMES[c_idx] if c_idx < len(CLASS_NAMES) else f"Clase {c_idx}"
        
        for pred_c_idx in range(num_classes):
            if pred_c_idx == c_idx:
                continue
                
            pred_name = CLASS_NAMES[pred_c_idx] if pred_c_idx < len(CLASS_NAMES) else f"Clase {pred_c_idx}"
            
            plt.plot(f_values, error_rates[:, pred_c_idx], marker='o', linewidth=2,color=COLOR_DICT[pred_c_idx], label=pred_name)
            
        plt.title(f'Error Distribution under Bias Variation - Target: {target_name}', fontweight='bold', fontsize=16)
        plt.xlabel(r'Bias Factor ($f$)')
        plt.ylim(0,0.15)
        plt.ylabel('False Negative Proportion')
        
        plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Misclassified as", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        
        plt.tight_layout()
        filename = f'error_distribution_stereo_c{c_idx}_{target_name}.pdf'
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
        
        print(f"Guardada gráfica: {filename}")


    if os.path.exists("./temp_logs/output.log"):
        os.remove("./temp_logs/output.log")
        os.rmdir("./temp_logs")
    
    print("Proceso finalizado")

if __name__ == "__main__":
    main()
