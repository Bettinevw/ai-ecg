import os, sys
from tensorflow import keras
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score

# Add preprocessing directory to work directory
dir_path        = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/model/')

from model_keras import build_ecg_hybrid, ResBlock

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"

name = 'stratified'

X_train = np.load(folder+"train_dataset_" +name+ ".npy")["ecg"].astype("float32")
X_val   = np.load(folder+"val_dataset_" + name + ".npy")["ecg"].astype("float32")
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_train = np.load(folder+"train_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
y_val   = np.load(folder+"val_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)

best_model_file = folder + "model_" + name + ".h5"

model = keras.models.load_model(best_model_file)

for split, X, y in [("train", X_train, y_train),
                    ("val",   X_val,   y_val),
                    ("test",  X_test,  y_test)]:
    y_prob = model.predict(X, batch_size=128).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print(split, "AUC:", roc_auc_score(y, y_prob),
          "Acc:", accuracy_score(y, y_pred),
          "Sen/Spe:", recall_score(y, y_pred),
          recall_score(y==0, y_pred==0))