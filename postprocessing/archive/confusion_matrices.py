# Confusion Matrices

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors

folder      = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name        = 'stratified'
test_file   = folder + "dataset_test_" + name + ".xlsx"
df          = pd.read_excel(test_file)
X_test      = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test      = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob      = np.load(output_file)["probs_test"].astype("float32")
thr         = np.load(output_file)["thr"].astype("float32")

# add model results to df
df["y_prob"] = y_prob 

# Timing
days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3])*days_per_year
time_labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"] = pd.cut(df['days_to_index'], bins=time_bins, labels=time_labels, right=False)
df["time_bin"] = pd.Categorical(df["time_bin"],
                                categories=time_labels[::-1],
                                ordered=True)

# Age bin
age_bins      = [17, 60, 70, 80, 200]
age_labels    = ["18–60", "60–70", "70–80", "≥80"]
df["age_bin"] = pd.cut(df["age_at_last_ecg"], bins=age_bins, labels=age_labels, right=False)

df["y_pred"] = (df["y_prob"] >= thr).astype(int)
df["y_true"] = y_test[df.index]


#%% Confusion matrices for each age group
df_age = df.groupby("age_bin")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Confusion matrix for each age group', fontsize=16)
count = 0.01
for name, group in df_age:
    ax = axes[int(np.round(count%2)), int(np.round(count/2))] 
    
    # cm = confusion_matrix(group['y_true'], group['y_pred'], labels=[0, 1])
    
    tp = len(group[(group['y_true'] == 1) & (group['y_pred'] == 1)])
    tn = len(group[(group['y_true'] == 0) & (group['y_pred'] == 0)])
    fp = len(group[(group['y_true'] == 0) & (group['y_pred'] == 1)])
    fn = len(group[(group['y_true'] == 1) & (group['y_pred'] == 0)])
    
    cm = np.array([[tp, fp],
                   [fn, tn]])

    # Maak een matrix van kleuren: groen voor true negative en true positive, rood voor de rest
    cell_colors = [['tab:green' if (i == j) else 'tab:red' for j in range(2)] for i in range(2)]

    # Plot de confusion matrix met handmatige kleuren
    for i in range(2):
        for j in range(2):
            color = cell_colors[i][j]
            ax.text(j + 0.5, 1.5 - i, str(cm[i, j]),
                    ha="center", va="center", color="black")
            ax.add_patch(plt.Rectangle((j, 1.5 - i-0.5), 1, 1, fill=True, color=cell_colors[i][j], alpha=0.7))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=12)
    ax.set_yticklabels(['Negative', 'Positive'], fontsize=12)
    ax.set_title(f'{name}')
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(False)

    count += 1

plt.tight_layout()
plt.show()

#%% Confusion matrices for each time group
df_time = df.groupby("time_bin")

fig, axes = plt.subplots(4, 2, figsize=(10, 10))
fig.suptitle('Confusion matrix for each time group', fontsize=16)
count = 0.01
for name, group in df_time:
    ax = axes[int(np.round(count%2)), int(np.round(count/2))] 
    print([np.round(count%2), np.round(count/2)])
    # cm = confusion_matrix(group['y_true'], group['y_pred'], labels=[0, 1])
    
    tp = len(group[(group['y_true'] == 1) & (group['y_pred'] == 1)])
    tn = len(group[(group['y_true'] == 0) & (group['y_pred'] == 0)])
    fp = len(group[(group['y_true'] == 0) & (group['y_pred'] == 1)])
    fn = len(group[(group['y_true'] == 1) & (group['y_pred'] == 0)])
    
    cm = np.array([[tp, fp],
                   [fn, tn]])

    # Maak een matrix van kleuren: groen voor true negative en true positive, rood voor de rest
    cell_colors = [['tab:green' if (i == j) else 'tab:red' for j in range(2)] for i in range(2)]

    # Plot de confusion matrix met handmatige kleuren
    for i in range(2):
        for j in range(2):
            color = cell_colors[i][j]
            ax.text(j + 0.5, 1.5 - i, str(cm[i, j]),
                    ha="center", va="center", color="black")
            ax.add_patch(plt.Rectangle((j, 1.5 - i-0.5), 1, 1, fill=True, color=cell_colors[i][j], alpha=0.7))

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Positive', 'Negative'], fontsize=12)
    ax.set_yticklabels(['Negative', 'Positive'], fontsize=12)
    ax.set_title(f'{name}')
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(False)

    count += 1

plt.tight_layout()
plt.show()