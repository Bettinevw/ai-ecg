import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add preprocessing directory to work directory
dir_path        = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/model/')

from model_keras import build_ecg_hybrid, ResBlock
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"

X_val   = np.load(folder+"val_dataset_stratified.npy")["ecg"].astype("float32")
X_test  = np.load(folder+"test_dataset_stratified.npy")["ecg"].astype("float32")
y_val   = np.load(folder+"val_outcome_stratified.npz")["outcome"].astype("int32").reshape(-1)
y_test  = np.load(folder+"test_outcome_stratified.npz")["outcome"].astype("int32").reshape(-1)
    
X_train = np.load(folder+"train_dataset_stratified.npy")["ecg"].astype("float32")
y_train = np.load(folder+"train_outcome_stratified.npz")["outcome"].astype("int32").reshape(-1)
best_model_file = folder + "model_stratified.h5"
output_file = folder + "prob_stratified.npz"
    
output_info = np.load(output_file)
probs_val = output_info['probs_val']
probs_test = output_info['probs_test']
thr = output_info['thr']
y_hat = output_info['y_hat']
auc = roc_auc_score(y_test, probs_test)

fpr, tpr, thresholds = roc_curve(y_test, probs_test)
auc = roc_auc_score(y_test, probs_test)
# Plotten
plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, label=f"ROC-curve (AUROC = {auc:.2f})", color='tab:purple', lw=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonaal
plt.xlabel("False Positive Rate", fontsize='large')
plt.ylabel("True Positive Rate", fontsize='large')
plt.title("ROC", fontsize='large')
plt.legend(loc="lower right", fontsize='large')
plt.grid(True)
plt.tight_layout()
plt.show()
