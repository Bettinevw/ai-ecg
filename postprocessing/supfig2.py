# performance model
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,               # Standaard lettergrootte
    'axes.titlesize': 14,          # Grootte van de assen-titels
    'axes.labelsize': 10,          # Grootte van de assen-labels
    'xtick.labelsize': 12,         # Grootte van de x-as ticklabels
    'ytick.labelsize': 12,         # Grootte van de y-as ticklabels
    'lines.linewidth': 3,          # Dikte van lijnen (bijv. medianen in violin plots)
    'grid.alpha': 0.7,             # Transparantie van de gridlijnen
})

#%% Dataset
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
manuscript_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Manuscript/Submission2/"
name = 'stratified'
color = 'tab:purple'

test_file   = folder + "dataset_test_" + name + ".xlsx"
y_test      = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob      = np.load(output_file)["probs_test"].astype("float32")
thr         = np.load(output_file)["thr"].astype("float32")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(4, 4))
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal


# Plot    
plt.plot(fpr, tpr, label=f"AUROC = {auc:.2f}", color=color)    
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([0,1])
plt.ylim([0,1])
plt.tight_layout()
plt.show()