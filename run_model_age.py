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

from model_keras_age import build_ecg_hybrid, ResBlock
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------- Folders ----------------------------------------------- #
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"

name = 'stratified'
X_train = np.load(folder+"train_dataset_" +name+ ".npy")["ecg"].astype("float32")
X_val   = np.load(folder+"val_dataset_" + name + ".npy")["ecg"].astype("float32")
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")

y_train = np.load(folder+"train_outcome_" + name + ".npz")["out_age"].astype("float32").reshape(-1)
y_val   = np.load(folder+"val_outcome_" + name + ".npz")["out_age"].astype("float32").reshape(-1)
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["out_age"].astype("float32").reshape(-1)

best_model_file = folder + "model_age_" + name + ".h5"
output_file = folder + "prob_age_"+name


# ------------------- 1. Model training ------------------------------------- #
# Bouw het basismodel
base = build_ecg_hybrid(input_length=X_train.shape[1], n_leads=X_train.shape[2], dropout=0.2)

# Gebruik alleen de bottleneck output voor age_pred
bottleneck = base.get_layer("bottleneck").output
age_pred = keras.layers.Dense(1, activation="linear", name="age_pred")(bottleneck)

# Definieer het model met alleen de age_pred output
model_age = keras.Model(inputs=base.input, outputs=age_pred, name="ECG_age_regressor")

# Compileer het model
model_age.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.Huber(delta=5.0),
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

# Fit het model
hist = model_age.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=4, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(best_model_file, monitor="val_mae", mode="min", save_best_only=True)
    ]
)





# model_age = build_ecg_hybrid(input_length=X_train.shape[1], 
#                         n_leads=X_train.shape[2], 
#                         dropout=0.2)
   
# model_age.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss=keras.losses.Huber(delta=5.0),
#     metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
# )

   
# hist = model_age.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=50, batch_size=128,
#     callbacks=[
#         keras.callbacks.EarlyStopping(monitor="val_age_pred_mae", mode="min", patience=8, 
#                                       restore_best_weights=True),
#         keras.callbacks.ReduceLROnPlateau(monitor="val_age_pred_mae", mode="min", factor=0.5,
#                                           patience=4, min_lr=1e-6),
#         keras.callbacks.ModelCheckpoint(best_model_file, monitor="val_age_pred_mae", 
#                                         mode="min", save_best_only=True)
#     ]
# )
