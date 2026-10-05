# --- SALIENCY MAPS ---

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add preprocessing directory to work directory
folder_software = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/software/"
sys.path.insert(0, folder_software + '/model/')
from model_keras import build_ecg_hybrid, ResBlock

# plt.close('all')

def saliency_map_tf(model, x, eps=1e-12):
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        p = model(x_tf, training=False)  # (B,1)

        # heel lichte clipping, zodat log niet explodeert
        p = tf.clip_by_value(p, eps, 1.0 - eps)

        # logit(p) = log(p) - log(1-p) maar numeriek stabieler:
        score = tf.math.log(p) - tf.math.log1p(-p)  # (B,1)

        score = tf.reduce_sum(score)

    grads = tape.gradient(score, x_tf)
    return tf.abs(grads).numpy()

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = 'stratified'

test_file = folder + "dataset_test_" + name + ".xlsx"
df      = pd.read_excel(test_file)
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)

name = 'age_stratified'
output_file = folder + "prob_"+name+".npz"
y_prob = np.load(output_file)["probs_test"].astype("float32")
# thr = np.load(output_file)["thr"].astype("float32")

best_model_file = folder + "model_" + name + ".h5"
best_model = keras.models.load_model(best_model_file,
                                     custom_objects={'ResBlock': ResBlock})

df = df.copy()
df["y_prob"] = y_prob 


# idx = int(np.argmax(y_prob))   # meest “positief”
# of:
idx = int(np.argmin(y_prob))   # meest “negatief”
lead = 1

X_ex = X_test[idx:idx+1].astype("float32")  # (N, timesteps, nr leads)
y_ex = y_test[idx]

sal = saliency_map_tf(best_model, X_ex)  # (N, timesteps, nr leads)
sal_T = sal[0,:,1]
ecg_T = X_ex[0,:,1]
T = len(ecg_T)
t = np.arange(T) / 250

# robuuste normalisatie (percentiel i.p.v. max) → voorkomt “alles donker”
s = sal_T.copy()
s = s / (np.percentile(s, 99) + 1e-12)
s = np.clip(s, 0, 1)

# Maak een achtergrond “image” van 1 rij, zodat imshow verticale banden geeft
bg = np.tile(s, (40, 1))  # 40 pixels hoog, puur voor look

fig, ax = plt.subplots(figsize=(12, 2.5))

# Blauwe banden (saliency) achter de ECG
ax.imshow(
    bg,
    aspect="auto",
    extent=[t[0], t[-1], np.min(ecg_T), np.max(ecg_T)],
    origin="lower",
    alpha=0.35,     # transparantie zoals in Jabbour
    cmap="Blues"    # licht→donker blauw
)

# Rode ECG lijn
ax.plot(t, ecg_T, color="red", linewidth=1.0)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Lead 1")
plt.tight_layout()
plt.show()

# lead = 0
# fig, ax1 = plt.subplots(figsize=(10,3))
# ax1.plot(X_ex[k,:,lead])
# ax1.set_ylabel("ECG")

# ax2 = ax1.twinx()
# s_norm = sal[k,:,lead] / (sal[k,:,lead].max() + 1e-12)
# ax2.plot(s_norm)
# ax2.set_ylabel("Saliency (0-1)")

# plt.title(f"Lead {lead} overlay (idx={idx[k]})")
# plt.tight_layout()
# plt.show()

# high = df["y_prob"].values > 0.9
# low  = df["y_prob"].values < 0.1

# def mean_saliency_for_mask(mask, batch=64):
#     all_mean = []
#     X = X_test[mask]
#     for i in range(0, len(X), batch):
#         sb = saliency_map_tf(best_model, X[i:i+batch].astype("float32"))
#         all_mean.append(sb.mean(axis=0))  # (T,C)
#     return np.mean(np.stack(all_mean, axis=0), axis=0)

# mean_hi = mean_saliency_for_mask(high)
# mean_lo = mean_saliency_for_mask(low)

# for title, M in [("Mean saliency | y_prob>0.9", mean_hi), ("Mean saliency | y_prob<0.1", mean_lo)]:
#     vmax = np.percentile(M, 99)
#     plt.figure(figsize=(10,3))
#     plt.imshow(M.T, aspect="auto", vmin=0, vmax=vmax)
#     plt.xlabel("time samples")
#     plt.ylabel("lead")
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()