import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add preprocessing directory to work directory
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/software/Submission2_EHJDH/"
sys.path.insert(0, folder + '/model/')
from model_keras import build_ecg_hybrid, ResBlock

# ----------------------------
# Saliency functions
# ----------------------------
def saliency_classifier_logit(model, x, eps=1e-12):
    """|d(logit(p))/dx| for binary classifier with sigmoid output p."""
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        p = model(x_tf, training=False)
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        score = tf.reduce_sum(tf.math.log(p) - tf.math.log1p(-p))  # sum over batch
    grads = tape.gradient(score, x_tf)
    return tf.abs(grads).numpy()

def saliency_regressor(model, x):
    """|d(y_pred)/dx| for regressor, per sample in batch."""
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        y = model(x_tf, training=False)  # Vorm: (batch_size, 1)
        # Bereken de gradient voor elke sample afzonderlijk
        # Gebruik y zelf (geen reduce_sum), omdat we de gradient per sample willen
        score = y  # Vorm: (batch_size, 1)
    grads = tape.gradient(score, x_tf)  # Vorm: (batch_size, input_shape)
    return tf.abs(grads).numpy()  # Absolute waarde per sample


def norm01(s):
    s = np.asarray(s, dtype=float)
    s = s / (np.percentile(s, 99) + 1e-12)
    return np.clip(s, 0, 1)

def overlay_single(ax, t, ecg, sal, title, cmap="Blues", ecg_color="red", alpha=0.35):
    s = norm01(sal)
    bg = np.tile(s, (40, 1))
    ax.imshow(bg, aspect="auto",
              extent=[t[0], t[-1], np.min(ecg), np.max(ecg)],
              origin="lower", alpha=alpha, cmap=cmap)
    ax.plot(t, ecg, color=ecg_color, linewidth=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])

def overlay_overlap(ax, t, ecg, sal_af, sal_age, sal_sex, title,
                    colors=("tab:blue", "gold", "tab:purple"),
                    alpha=0.22):
    """
    Overlap panel: show three saliency bands with different colors + black ECG.
    """
    y_min, y_max = float(np.min(ecg)), float(np.max(ecg))
    s_af  = norm01(sal_af)
    s_age = norm01(sal_age)
    s_sex = norm01(sal_sex)

    # Build RGBA images (40 x T x 4)
    H = 40
    def to_rgba(s, color_name):
        rgba = np.zeros((H, len(s), 4), dtype=float)
        # use matplotlib to convert named color to rgb
        rgb = np.array(plt.matplotlib.colors.to_rgb(color_name), dtype=float)
        rgba[..., :3] = rgb[None, None, :]
        rgba[..., 3] = s[None, :] * alpha
        return rgba

    im_af  = to_rgba(s_af,  colors[0])
    im_age = to_rgba(s_age, colors[1])
    im_sex = to_rgba(s_sex, colors[2])

    ax.imshow(im_af, aspect="auto",
              extent=[t[0], t[-1], y_min, y_max], origin="lower")
    ax.imshow(im_age, aspect="auto",
              extent=[t[0], t[-1], y_min, y_max], origin="lower")
    ax.imshow(im_sex, aspect="auto",
              extent=[t[0], t[-1], y_min, y_max], origin="lower")

    ax.plot(t, ecg, color="black", linewidth=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_yticks([])

# Files
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = 'stratified'

test_file = folder + "dataset_test_" + name + ".xlsx"
df      = pd.read_excel(test_file)
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)

output_file_age = folder + "prob_age_stratified.npz"
model_file_age = folder + "model_age_stratified.h5"
model_age = keras.models.load_model(model_file_age,
                                      custom_objects={'ResBlock': ResBlock})
output_file_af = folder + "prob_stratified.npz"
model_file_af = folder + "model_stratified.h5"
model_af = keras.models.load_model(model_file_af,
                                      custom_objects={'ResBlock': ResBlock})
output_file_sex = folder + "prob_sex_stratified.npz"
model_file_sex = folder + "model_sex_stratified.h5"
model_sex = keras.models.load_model(model_file_sex,
                                      custom_objects={'ResBlock': ResBlock})

y_prob_af = np.load(output_file_af)["probs_test"].astype("float32")
# thr = np.load(output_file)["thr"].astype("float32")



# df = df.copy()
# df["y_prob"] = y_prob 


# ----------------------------
# Choose your leads (set indices correctly!)
# ----------------------------
LEAD_II_IDX = 1   # <-- set to your Lead II index
LEAD_V1_IDX = 2   # <-- set to your V1 index

FS = 250  # adjust if your sampling rate differs
IDX = int(np.argsort(y_prob_af)[0])  # patient with highest AF prob (you already have y_prob_af)
# IDX = int(np.argmin(y_prob_af))
# ----------------------------
# Prepare sample
# ----------------------------
X_ex = X_test[IDX:IDX+1].astype("float32")  # (1,T,C)
T = X_ex.shape[1]
t = np.arange(T) / FS

# Compute saliency maps for this sample
sal_af  = saliency_classifier_logit(model_af,  X_ex)   # (1,T,C)
sal_age = saliency_regressor(model_age, X_ex)          # (1,T,C)
sal_sex = saliency_classifier_logit(model_sex, X_ex)   # (1,T,C)

# ----------------------------
# Plot 4 rows x 2 cols
# ----------------------------
lead_list = [(LEAD_II_IDX, "Lead II"), (LEAD_V1_IDX, "Lead V1")]

fig, axs = plt.subplots(4, 2, figsize=(14, 9), sharex=True)

for col, (ld, ld_name) in enumerate(lead_list):
    ecg = X_ex[0, :, ld]

    # Row 1: AF
    overlay_single(
        axs[0, col], t, ecg, sal_af[0, :, ld],
        title=f"AF saliency — {ld_name}",
        cmap="Blues", ecg_color="black", alpha=0.35
    )

    # Row 2: Age
    overlay_single(
        axs[1, col], t, ecg, sal_age[0, :, ld],
        title=f"Age saliency — {ld_name}",
        cmap="Oranges", ecg_color="black", alpha=0.35
    )

    # Row 3: Sex
    overlay_single(
        axs[2, col], t, ecg, sal_sex[0, :, ld],
        title=f"Sex saliency — {ld_name}",
        cmap="Purples", ecg_color="black", alpha=0.35
    )

    # Row 4: overlap (color-coded)
    overlay_overlap(
        axs[3, col], t, ecg,
        sal_af[0, :, ld], sal_age[0, :, ld], sal_sex[0, :, ld],
        title=f"Overlap (AF=blue, Age=yellow, Sex=purple) — {ld_name}",
        colors=("tab:blue", "tab:orange", "tab:purple"),
        alpha=0.22
    )

# axis labels
for c in range(2):
    axs[3, c].set_xlabel("Time (s)")
for r in range(4):
    axs[r, 0].set_ylabel(["AF", "Age", "Sex", "Overlap"][r])

plt.tight_layout()
plt.show()
