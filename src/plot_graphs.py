import os
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- CONFIGURATION ---
WANDB_PATH = "mariogutierrezlopez-upm/MultiPIE_Stereotypical_All_Classes_correct" 
EMOTIONS = {0: "Neutral", 1: "Smile", 2: "Surprise", 3: "Squint", 4: "Disgust", 5: "Scream"}
PALETTE = {"Male": "#1D3557", "Female": "#E63946"}
BASE_OUTPUT_DIR = "plots_output" # All folders will be created inside this main directory

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

api = wandb.Api()
runs = api.runs(WANDB_PATH)

data = []

print("⬇️ Fetching runs from WandB...")

for run in runs:
    if run.state != "finished":
        continue

    # Regex to distinguish between the two experiment types
    stereo_match = re.search(r"Stereotipical_bias_c(\d+)_f(\d+\.?\d*)", run.name)
    repres_match = re.search(r"Repres_bias_f(\d+\.?\d*)", run.name)

    if stereo_match:
        exp_type = "Stereotypical"
        c_target = int(stereo_match.group(1)) # The experiment focus
        f_val = float(stereo_match.group(2))
    elif repres_match:
        exp_type = "Representational"
        c_target = "Representational" # Identifier for the repres runs
        f_val = float(repres_match.group(1))
    else:
        continue

    # Loop through every class to check its recall within the current run
    for eval_idx, eval_name in EMOTIONS.items():
        m_key = f"test_recall_class_{eval_idx}_MALE"
        f_key = f"test_recall_class_{eval_idx}_FEMALE"
        
        m_recall = run.summary.get(m_key) or run.summary.get(f"{m_key}_epoch")
        f_recall = run.summary.get(f_key) or run.summary.get(f"{f_key}_epoch")

        if m_recall is not None and f_recall is not None:
            m_val, f_val_rec = float(m_recall), float(f_recall)
            entry = {
                "f": f_val,
                "Exp_Type": exp_type,
                "Target_Experiment": c_target,
                "Eval_Class_Name": eval_name,
                "Eval_Class_ID": eval_idx,
                "Equity Gap": f_val_rec - m_val
            }
            # Append Male
            data.append({**entry, "Recall": m_val, "Gender": "Male"})
            # Append Female
            data.append({**entry, "Recall": f_val_rec, "Gender": "Female"})

df = pd.DataFrame(data)

# --- PLOTTING LOGIC ---

def save_plot(df_plot, title, y_col, folder_path, filename, is_gap=False, hue=None):
    # Ensure the target directory exists
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    
    plt.figure(figsize=(10, 6))
    
    if is_gap:
        # Plotting the gap directly (no hue needed since it is F-M)
        sns.lineplot(data=df_plot, x="f", y=y_col, markers="o", dashes=False, 
                     linewidth=3, markersize=10, color="#1D3557")
        plt.axhline(0, color='black', linestyle='--', alpha=0.7)
        plt.ylim(-0.8, 0.8) # Bounding equity gap between -0.8 and 0.8
        plt.ylabel("Equity Gap (F - M)")
    else:
        # Plotting Recall with Gender hue
        sns.lineplot(data=df_plot, x="f", y=y_col, hue=hue, markers=True, 
                     linewidth=3, palette=PALETTE, markersize=10)
        plt.ylim(0, 1.05)
        plt.ylabel(y_col)

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(r"Bias Factor ($f$)")
    plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    if not is_gap and hue:
        plt.legend(title="Gender", frameon=True)
        
    sns.despine()
    plt.savefig(filepath, format="svg", bbox_inches="tight")
    plt.close()

# 1. Generate Stereotypical Plots (Grouped by Target Experiment)
for target_idx, target_name in EMOTIONS.items():
    folder_path = os.path.join(BASE_OUTPUT_DIR, f"Target_Class_{target_idx}_{target_name}")
    exp_df = df[(df["Exp_Type"] == "Stereotypical") & (df["Target_Experiment"] == target_idx)]
    
    if exp_df.empty:
        continue
        
    print(f"📁 Generating plots inside: {folder_path}")
    
    for eval_idx, eval_name in EMOTIONS.items():
        eval_subset = exp_df[exp_df["Eval_Class_ID"] == eval_idx]
        
        if not eval_subset.empty:
            # 1. Gender Recall Plot - Formato: ID_nombre_recall.svg
            save_plot(
                eval_subset, 
                f"Target Exp {target_name} ($c={target_idx}$) | Recall Variation on {eval_name}", 
                "Recall", 
                folder_path,
                f"{eval_idx}_{eval_name.lower()}_recall.svg", 
                hue="Gender"
            )
            
            # 2. Equity Gap Plot - Formato: ID_nombre_gap.svg
            save_plot(
                eval_subset, 
                f"Target Exp {target_name} ($c={target_idx}$) | Equity Gap on {eval_name}", 
                "Equity Gap", 
                folder_path,
                f"{eval_idx}_{eval_name.lower()}_gap.svg", 
                is_gap=True
            )

# 2. Generate Representational Plots
repres_folder = os.path.join(BASE_OUTPUT_DIR, "Representational_Experiment")
repres_df = df[df["Exp_Type"] == "Representational"]

if not repres_df.empty:
    print(f"📁 Generating plots inside: {repres_folder}")
    for eval_idx, eval_name in EMOTIONS.items():
        eval_subset = repres_df[repres_df["Eval_Class_ID"] == eval_idx]
        
        if not eval_subset.empty:
            # 1. Gender Recall Plot
            save_plot(
                eval_subset, 
                f"Representational Exp | Recall Variation on {eval_name}", 
                "Recall", 
                repres_folder,
                f"{eval_idx}_{eval_name.lower()}_recall.svg", 
                hue="Gender"
            )
            
            # 2. Equity Gap Plot
            save_plot(
                eval_subset, 
                f"Representational Exp | Equity Gap on {eval_name}", 
                "Equity Gap", 
                repres_folder,
                f"{eval_idx}_{eval_name.lower()}_gap.svg", 
                is_gap=True
            )

print("🚀 Generation complete. Check the 'plots_output' directory for the organized SVG files.")