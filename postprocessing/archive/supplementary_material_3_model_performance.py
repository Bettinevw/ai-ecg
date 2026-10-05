import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add preprocessing directory to work directory
dir_path        = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/model/')

from model_keras_age import build_ecg_hybrid, ResBlock
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report, confusion_matrix, roc_auc_score

plt.rcParams.update({
    'font.size': 12,               # Standaard lettergrootte
    'axes.titlesize': 14,          # Grootte van de assen-titels
    'axes.labelsize': 12,          # Grootte van de assen-labels
    'xtick.labelsize': 10,         # Grootte van de x-as ticklabels
    'ytick.labelsize': 10,         # Grootte van de y-as ticklabels
    'lines.linewidth': 2,          # Dikte van lijnen (bijv. medianen in violin plots)
    'grid.alpha': 0.7,             # Transparantie van de gridlijnen
})

#%% ------------------- Folders ----------------------------------------------- #
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"

name = 'stratified'
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["out_age"].astype("float32").reshape(-1)

model_age_file = folder + "model_age_" + name + ".h5"
output_age_file = folder + "prob_age_"+name

best_model_age = keras.models.load_model(model_age_file,
                                     custom_objects={'ResBlock': ResBlock})

# Predict continuous values
y_pred_test = best_model_age.predict(X_test, verbose=0).squeeze()

# Core metrics
mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

print(f"MAE (years):  {mae:.2f}")
print(f"RMSE (years): {rmse:.2f}")
print(f"R2 score:     {r2:.3f}")

# Clinically interpretable metric
within_5y = np.mean(np.abs(y_pred_test - y_test) <= 5.0)
within_10y = np.mean(np.abs(y_pred_test - y_test) <= 10.0)

print(f"Within ±5 years:  {within_5y*100:.1f}%")
print(f"Within ±10 years: {within_10y*100:.1f}%")

# Save outputs
np.savez(
    output_age_file,
    y_pred_test=y_pred_test,
    y_true_test=y_test
)

# Bland-Altman plot
plt.figure(figsize=(8, 6))
diff = y_pred_test - y_test
md = np.mean(diff)
sd = np.std(diff, axis=0)
plt.scatter(y_test, diff, alpha=0.5, color='tab:purple')
plt.axhline(md, color='gray', linestyle='--')
plt.axhline(md + 1.96*sd, color='gray', linestyle=':')
plt.axhline(md - 1.96*sd, color='gray', linestyle=':')
plt.title("Age prediction")
plt.xlabel("True Age [years]")
plt.ylabel("Difference (Predicted - True)")
plt.xlim([0, 100])
plt.ylim([-60, 60])

#%%
output_sex_file = folder + "prob_sex_"+name+".npz"
y_prob_sex      =  np.load(output_sex_file)["probs_test"].astype("float32")
thr         = np.load(output_sex_file)["thr"].astype("float32")
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["out_sex"].astype("float32").reshape(-1)

y_pred = (y_prob_sex >= thr).astype(int)

fpr, tpr, thresholds = roc_curve(y_test, y_prob_sex)
auc = roc_auc_score(y_test, y_prob_sex)

# Plotten
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, label=f"ROC-curve (AUC = {auc:.2f})", color='tab:purple', lw=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonaal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid(True)
plt.tight_layout()
plt.show()