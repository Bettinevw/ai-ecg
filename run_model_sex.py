###############################################################################
#
# Machine learning model based on Attia et al. 2019
#
###############################################################################

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


# ------------------- Folders ----------------------------------------------- #
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"

name = 'stratified'
X_train = np.load(folder+"train_dataset_" +name+ ".npy")["ecg"].astype("float32")
X_val   = np.load(folder+"val_dataset_" + name + ".npy")["ecg"].astype("float32")
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")

# MALE=0; FEMALE=1
y_train = np.load(folder+"train_outcome_" + name + ".npz")["out_sex"].astype("int32").reshape(-1)
y_val   = np.load(folder+"val_outcome_" + name + ".npz")["out_sex"].astype("int32").reshape(-1)
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["out_sex"].astype("int32").reshape(-1)

best_model_file = folder + "model_sex_" + name + ".h5"
output_file = folder + "prob_sex_"+name

base = build_ecg_hybrid(input_length=X_train.shape[1], 
                        n_leads=X_train.shape[2], 
                        dropout=0.2)

# train alleen op y_pred
model = keras.Model(base.input, 
                  base.get_layer("y_pred").output, 
                  name="ECGHybrid_gender_classifier")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
              keras.metrics.AUC(name="auc_roc", curve="ROC")]
)

classes = np.array([0, 1], dtype=np.int32)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)} # {0: 0.625, 1: 2.5}

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=128,
    class_weight=class_weight,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=8, 
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.5,
                                          patience=4, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(best_model_file, monitor="val_auc_roc", 
                                        mode="max", save_best_only=True)
    ]
)
