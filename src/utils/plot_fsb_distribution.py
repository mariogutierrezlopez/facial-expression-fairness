"""
Analiza y compara la distribución de iluminación (FSB) en los datasets AffectNet, RAF-DB y AffWild2.
Genera histogramas individuales, una comparativa de densidad (KDE) conjunta y una tabla de 
estadísticas descriptivas (media y desviación típica) para evaluar el sesgo lumínico.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def process_and_plot():
    datasets_info = {
        'AffectNet': [
            "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/train_pose_fsb.csv",
            "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/val_pose_fsb.csv",
            "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/no_human_annotated_pose_fsb.csv"
        ],
        'RAF-DB': [
            "/home12TB1/database/recognition/faces/RAF-DB/img2pose_ann/train_pose_fsb.csv",
            "/home12TB1/database/recognition/faces/RAF-DB/img2pose_ann/test_pose_fsb.csv"
        ],
        'AffWild2': [
            "/home12TB1/database/recognition/faces/affwild2/img2pose_ann/affwild2_fsb.csv",
            "/home12TB1/database/recognition/faces/affwild2/img2pose_ann/affwild2_val_fsb.csv"
        ]
    }

    combined_dfs = {}
    stats_list = []

    # 1. Carga y procesado
    for name, files in datasets_info.items():
        temp_dfs = []
        for f in files:
            if os.path.exists(f):
                df = pd.read_csv(f)
                df.columns = [c.upper() for c in df.columns]
                if 'FSB' in df.columns:
                    temp_dfs.append(df[df['FSB'] > 0])
        
        if temp_dfs:
            combined = pd.concat(temp_dfs)
            combined_dfs[name] = combined
            
            stats_list.append({
                'Dataset': name,
                'Media': round(combined['FSB'].mean(), 2),
                'Desv. Típica': round(combined['FSB'].std(), 2),
                'Muestras': len(combined)
            })

            plt.figure()
            sns.histplot(combined['FSB'], kde=True, color='skyblue', stat="density")
            plt.title(f'Distribución FSB - {name}')
            plt.savefig(f'distribucion_fsb_{name.lower()}.pdf')
            plt.close()

    plt.figure(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (name, df), color in zip(combined_dfs.items(), colors):
        sns.kdeplot(data=df['FSB'], label=name, fill=True, alpha=0.2, color=color, linewidth=3)

    plt.title('Comparativa de Face Skin Brightness (FSB)', fontsize=18)
    plt.xlabel('Valor FSB', fontsize=14)
    plt.ylabel('Densidad', fontsize=14)
    plt.xlim(0, 255)
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribucion_fsb.pdf', dpi=300)
    
    df_stats = pd.DataFrame(stats_list)
    print("\n" + "="*50)
    print("ESTADÍSTICAS DESCRIPTIVAS DEL FSB")
    print("="*50)
    print(df_stats.to_string(index=False))
    print("="*50)
    
    return df_stats

if __name__ == "__main__":
    process_and_plot()