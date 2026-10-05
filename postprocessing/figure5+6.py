import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.interpolate import interp1d, CubicSpline
from scipy.interpolate import PchipInterpolator
import scipy.stats as stats


plt.rcParams.update({
    'font.size': 12,               #
    'axes.titlesize': 14,          # 
    'axes.labelsize': 12,          #
    'xtick.labelsize': 10,         # 
    'ytick.labelsize': 10,         # 
    'lines.linewidth': 2,          # 
    'grid.alpha': 0.7,             # 
})

# Add preprocessing directory to work directory
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/software/Submission2_EHJDH/"
sys.path.insert(0, folder + '/model/')
from model_keras import build_ecg_hybrid, ResBlock

# ----------------------------
# Saliency functions
# ----------------------------
def normalize_zscore(signal):
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    return (signal - mean) / (std + 1e-8)  # prevent dividing by 0

def normalize_minmax(signal):
    min_val = np.min(signal, axis=1, keepdims=True)  # Minimum per lead
    max_val = np.max(signal, axis=1, keepdims=True)  # Maximum per lead
    return (signal - min_val) / (max_val - min_val + 1e-8)  # prevent dividing by 0


def normalize_max(signal):
    max_abs = np.max(np.abs(signal), axis=1, keepdims=True)  # Maximal absolute value per patient and lead
    return signal / (max_abs + 1e-8)  # Scale to [-1, 1]

def percentile_normalize(grads, low=1, high=99):
    p_low = np.percentile(grads, low)
    p_high = np.percentile(grads, high)
    return (grads - p_low) / (p_high - p_low + 1e-8)


def saliency_classifier_logit(model, x, eps=1e-12):
    """|d(logit(p))/dx| for binary classifier with sigmoid output p."""
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        p = model(x_tf, training=False)
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        score = tf.math.log(p) - tf.math.log1p(-p)  # sigmoid function
    grads = tape.gradient(score, x_tf)
    return tf.abs(grads).numpy()


def saliency_regressor(model, x):
    """|d(y_pred)/dx| for regressor, per sample in batch."""
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    # Compute the gradient for every sample seperately
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        y = model(x_tf, training=False)  # Shape: (batch_size, 1)
    grads = tape.gradient(y, x_tf)  # Shape: (batch_size, input_shape)
    return tf.abs(grads).numpy()  # Absolute value per sample



def get_rpeaks(signal, fs=250, lead_idx=0):
    """
    signal: (n_leads, n_samples)
    lead_idx: welke lead te gebruiken voor R-peak detectie
    """
    sig = signal[:, lead_idx]
    signals, info = nk.ecg_peaks(sig, method = 'kalidas2017', sampling_rate=fs, 
                                 correct_artifacts=False, show=False)
    
    r_peaks = info['ECG_R_Peaks']
    
    return r_peaks


def compute_median_beat(signal, sal, r_peaks, fs=250):
    n_samples, n_leads = signal.shape
    
    rr_intervals    = np.diff(r_peaks)
    rrmed           = np.median(rr_intervals)
    pre_samples     = int(0.45*rrmed)
    post_samples    = int(0.55*rrmed)
    beat_len        = pre_samples + post_samples

    beats = []
    beats_sal = []
    for r in r_peaks:
        start = r - pre_samples
        end   = r + post_samples
        if start < 0 or end > n_samples:
            continue
        
        beat = signal[start:end, :]  # Shape: (beat_len, n_leads)
        beat_sal = sal[start:end, :] # Shape: (beat_len, n_leads)

        if beat.shape[0] == beat_len:  # Check if every beat has the same length
            beats.append(beat)
            beats_sal.append(beat_sal)

    # Stack de lijsten naar NumPy arrays
    if len(beats) > 0:
        beats_array     = np.stack(beats, axis=0)      # shape: (nr_beats, beat_len, n_leads)
        beats_sal_array = np.stack(beats_sal, axis=0) # shape: (nr_beats, beat_len, n_leads)

        # Bereken de median over de beats
        median_beat     = np.median(beats_array, axis=0)      # shape: (beat_len, n_leads)
        median_beat_sal = np.median(beats_sal_array, axis=0) # shape: (beat_len, n_leads)
    else:
        median_beat     = np.zeros((beat_len, n_leads))  # Default output als er geen beats zijn
        median_beat_sal = np.zeros((beat_len, n_leads))

    return median_beat, median_beat_sal


def plot_af_sal(groups_ecg, groups_sal, LEAD_II_IDX, vmin, vmax):
    
    fig, ax = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle('AF prediction model')
    
    subgroups = ['Control', 'AF', 'Female', 'Male', '18-60', '≥80']
    
    for subgroup in range(3):
        # print('subgroup',subgroup)
        
        ecg_group = groups_ecg[subgroup]
        
        
        for row in range(1):
            # print('row', row) # type model        
            
            sal_group = groups_sal[row][subgroup]
            sal_group = groups_sal[row][subgroup]
            sal_group = groups_sal[row][subgroup]
            
            col = 0
            for key in ecg_group:
                # print('col', col)
                mean_ecg    = np.mean(ecg_group[key], axis=0)
                std_ecg     = np.std(ecg_group[key], axis=0)
                mean_sal    = np.mean(sal_group[key], axis=0)

                rel_time    = np.arange(0, np.shape(mean_ecg)[0])/(np.shape(mean_ecg)[0]-1)
                
                # Plot ECG
                ax[subgroup, col].plot(rel_time, mean_ecg[:, LEAD_II_IDX], "--", color='black', label=key)
                ax[subgroup, col].fill_between(
                    rel_time,
                    mean_ecg[:, LEAD_II_IDX] - std_ecg[:, LEAD_II_IDX],
                    mean_ecg[:, LEAD_II_IDX] + std_ecg[:, LEAD_II_IDX],
                    color='black', alpha=0.3, label="±1 Std Dev"
                )
    
                # Plot Saliency
                bg = np.tile(mean_sal[:, LEAD_II_IDX], (40, 1))
                im = ax[subgroup, col].imshow(
                    bg, aspect="auto",
                    extent=[0, 1, -1.5, 1.5],
                    origin="lower", cmap='Purples', vmin=vmin, vmax=vmax)
        
                ax[subgroup, col].set_ylim([-1.5, 1.5])

                ax[subgroup, col].set_title(f"{subgroups[col+subgroup*2]}")  
                               
                if col == 0:
                    ax[subgroup, col].set_ylabel(r"Normalized ECG amplitude [mV]")
                    
                if subgroup == 2: 
                    ax[subgroup, col].set_xlabel('Cardiac cycle time [-]')
                    
                
                col += 1
            
            mean_ecg_1 = np.mean(ecg_group[list(ecg_group.keys())[0]], axis=0)
            mean_ecg_2 = np.mean(ecg_group[list(ecg_group.keys())[1]], axis=0)
            mean_ecg_3 = mean_ecg_1 - mean_ecg_2
            
            mean_sal_1  = np.mean(sal_group[list(ecg_group.keys())[0]], axis=0)
            mean_sal_2  = np.mean(sal_group[list(ecg_group.keys())[1]], axis=0)
            mean_sal_3  = mean_sal_1 - mean_sal_2
            
            print(f'max diff saliency {np.max(mean_sal_3[:, LEAD_II_IDX])}')
            print(f'min diff salience {np.min(mean_sal_3[:, LEAD_II_IDX])}')
            u_stat, p_value_u = stats.mannwhitneyu(mean_sal_1[np.argmax(mean_sal_3[:,LEAD_II_IDX]), LEAD_II_IDX], 
                                                   mean_sal_2[np.argmax(mean_sal_3[:,LEAD_II_IDX]), LEAD_II_IDX])
            print(p_value_u)

            # ax[subgroup, 2].plot(rel_time, mean_ecg_1[:, LEAD_II_IDX], "--", color='tab:orange', label="Female")
            # ax[subgroup, 2].plot(rel_time, mean_ecg_2[:, LEAD_II_IDX], "--", color='tab:blue', label="Male")
            ax[subgroup, 2].plot(rel_time, mean_ecg_3[:, LEAD_II_IDX], "--", color='black', label="Difference")
            
            # Plot Saliency
            bg = np.tile(mean_sal_3[:, LEAD_II_IDX], (40, 1))
            im2 = ax[subgroup, 2].imshow(
                bg, aspect="auto",
                extent=[0, 1, -1.5, 1.5],
                origin="lower", cmap='RdBu_r', vmin=-vmax/4, vmax=vmax/4)#np.max([mean_sal_1[:, LEAD_II_IDX],mean_sal_2[:, LEAD_II_IDX]]))
            if subgroup == 0:
                title = 'Control - AF'
            elif subgroup == 1:
                title = 'Female - Male'
            else:
                title = '18-60 - ≥80'
            
            ax[subgroup, 2].set_title(title)
    
    ax[2,2].set_xlabel('Cardiac cycle time [-]')
        
    cbar_ax1 = fig.add_axes([0.83, 0.62, 0.02, 0.3])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax1, label="Normalized Saliency")

    cbar_ax2 = fig.add_axes([0.9, 0.62, 0.02, 0.3])  # [left, bottom, width, height]
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, label="Saliency Difference")
    cbar2.set_ticks([-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25])
        
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.show()



    
def plot_model_sal(ecg_1, ecg_2, sal_1, sal_2, LEAD_II_IDX, model_name, 
                   name_1, name_2, vmin, vmax):
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    
    if name_1 == 'Female':
        fig.suptitle('Sex classification model')
        ax[2].set_title('Female - Male')
    else:
        fig.suptitle('Age regression model')
        ax[2].set_title('18-60 - ≥80')
       
    mean_ecg_1  = np.mean(ecg_1, axis=0)
    std_ecg_1   = np.std(ecg_1, axis=0)
    mean_sal_1  = np.mean(sal_1, axis=0)
    
    mean_ecg_2  = np.mean(ecg_2, axis=0)
    std_ecg_2   = np.std(ecg_2, axis=0)
    mean_sal_2  = np.mean(sal_2, axis=0)
    
    mean_ecg_3 = mean_ecg_1 - mean_ecg_2
    mean_sal_3 = mean_sal_1 - mean_sal_2
    print(np.max(mean_sal_3[:, LEAD_II_IDX]))
    print(np.min(mean_sal_3[:, LEAD_II_IDX]))

    rel_time = np.arange(0, np.shape(mean_ecg_1)[0])/(np.shape(mean_ecg_1)[0]-1)
                
    # Plot ECG
    ax[0].plot(rel_time, mean_ecg_1[:, LEAD_II_IDX], "--", color='black', label=name_1)
    ax[0].fill_between(
        rel_time,
        mean_ecg_1[:, LEAD_II_IDX] - std_ecg_1[:, LEAD_II_IDX],
        mean_ecg_1[:, LEAD_II_IDX] + std_ecg_1[:, LEAD_II_IDX],
        color='black', alpha=0.3, label="±1 Std Dev"
    )

    # Plot Saliency
    bg = np.tile(mean_sal_1[:, LEAD_II_IDX], (40, 1))
    im = ax[0].imshow(
        bg, aspect="auto",
        extent=[0, 1, -1.5, 1.5],
        origin="lower", cmap='Purples',vmin=vmin, vmax=vmax)#np.max([mean_sal_1[:, LEAD_II_IDX],mean_sal_2[:, LEAD_II_IDX]]))
    ax[0].set_ylim([-1.5, 1.5])

    ax[0].set_title(name_1)  
    ax[0].set_xlabel('Cardiac cycle time [-]')
    
    ax[0].set_ylabel(r"Normalized ECG amplitude [mV]")
    
    ax[0].annotate(f"Model: {model_name}" , 
                          xy=(0, 0.05), xycoords="axes fraction", 
         xytext=(-90, 0), textcoords='offset pixels',
         fontsize=15, rotation="vertical")
    
    ax[1].set_title(name_2) 
    ax[1].plot(rel_time, mean_ecg_2[:, LEAD_II_IDX], "--", color='black', label=name_2)
    ax[1].fill_between(
        rel_time,
        mean_ecg_2[:, LEAD_II_IDX] - std_ecg_2[:, LEAD_II_IDX],
        mean_ecg_2[:, LEAD_II_IDX] + std_ecg_2[:, LEAD_II_IDX],
        color='black', alpha=0.3, label="±1 Std Dev"
    )

    # Plot Saliency
    bg = np.tile(mean_sal_2[:, LEAD_II_IDX], (40, 1))
    ax[1].imshow(
        bg, aspect="auto",
        extent=[0, 1, -1.5, 1.5],
        origin="lower", cmap='Purples', vmin=vmin, vmax=vmax)#np.max([mean_sal_1[:, LEAD_II_IDX],mean_sal_2[:, LEAD_II_IDX]]))
    ax[1].set_xlabel('Cardiac cycle time [-]')
    # ax[2].plot(rel_time, mean_ecg_1[:, LEAD_II_IDX], "--", color='tab:orange', label="Female")
    # ax[2].plot(rel_time, mean_ecg_2[:, LEAD_II_IDX], "--", color='tab:blue', label="Male")
    ax[2].plot(rel_time, mean_ecg_3[:, LEAD_II_IDX], "--", color='black', label="Difference")
    ax[2].set_xlabel('Cardiac cycle time [-]')
    # Plot Saliency
    bg = np.tile(mean_sal_3[:, LEAD_II_IDX], (40, 1))
    im2 = ax[2].imshow(
        bg, aspect="auto",
        extent=[0, 1, -1.5, 1.5],
        origin="lower", cmap='RdBu_r', vmin=-vmax/4, vmax=vmax/4)#np.max([mean_sal_1[:, LEAD_II_IDX],mean_sal_2[:, LEAD_II_IDX]]))
    
    cbar_ax1 = fig.add_axes([0.83, 0.1, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax1, label="Normalized Saliency")

    cbar_ax2 = fig.add_axes([0.9, 0.1, 0.02, 0.7])  # [left, bottom, width, height]
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, label="Saliency Difference")
    cbar2.set_ticks([-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25])
        
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    plt.show()


#%% Load data
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
out_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Results/Submission2_EHJDH/"
name = 'stratified'

test_file = folder + "dataset_test_" + name + ".xlsx"
df      = pd.read_excel(test_file)
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)

model_file_af = folder + "model_stratified.h5"
model_af = keras.models.load_model(model_file_af,
                                      custom_objects={'ResBlock': ResBlock})

model_file_sex = folder + "model_sex_stratified.h5"
model_sex = keras.models.load_model(model_file_sex,
                                      custom_objects={'ResBlock': ResBlock})

model_file_age = folder + "model_age_stratified.h5"
model_age = keras.models.load_model(model_file_age,
                                      custom_objects={'ResBlock': ResBlock})

#%% Run saliency
# ----------------------------
# Choose your leads 
# ----------------------------
LEAD_II_IDX = 1   # <-- set to your Lead II index
LEAD_V1_IDX = 2   # <-- set to your V1 index

FS = 250  # adjust if your sampling rate differs

#%% Run saliency
# ----------------------------
# Prepare sample
# ----------------------------
sals_af     = []
sals_sex    = []
sals_age    = []
for i_pat in range(X_test.shape[0]):
    if i_pat%100 == 0:
        print(i_pat)
    # Use 8 leads as input
    X_ex = X_test[i_pat:i_pat+1, :, :].astype("float32")  # shape: (1, input_length, 8)
    # Bereken saliency voor alle leads
    sal_af = saliency_classifier_logit(model_af, X_ex)  # shape: (1, input_length, 8)
    # Bereken saliency voor alle leads
    sal_sex = saliency_classifier_logit(model_sex, X_ex)  # shape: (1, input_length, 8)
    sal_age = saliency_regressor(model_age, X_ex)  
    # Save saliencies
    sals_af.append(sal_af[0, :, :])
    sals_sex.append(sal_sex[0, :, :])
    sals_age.append(sal_age[0, :, :])
       
np.savez(out_folder+'saliency_'+name, saliency_af=sals_af, saliency_sex=sals_sex, saliency_age=sals_age)

#%% Compute median beats
sals_af     = np.load(out_folder+'saliency_'+name+'.npz', allow_pickle=True)['saliency_af']
sals_sex    = np.load(out_folder+'saliency_'+name+'.npz', allow_pickle=True)['saliency_sex']
sals_age    = np.load(out_folder+'saliency_'+name+'.npz', allow_pickle=True)['saliency_age']

median_signals = {}
max_length = 0
for i_pat in range(X_test.shape[0]):
    if i_pat%100 == 0:
        print(i_pat)
        
    # Median beat + saliency
    r_peaks = get_rpeaks(X_test[i_pat,:,:], fs=FS, lead_idx=LEAD_II_IDX)

    # Gemiddelde beat
    median_beat, median_beat_sal_af = compute_median_beat(X_test[i_pat,:,:], 
                                                          sals_af[i_pat,:,:], r_peaks, fs=FS)
    _, median_beat_sal_sex = compute_median_beat(X_test[i_pat,:,:], 
                                                  sals_sex[i_pat,:,:], r_peaks, fs=FS)
    _, median_beat_sal_age = compute_median_beat(X_test[i_pat,:,:], 
                                                  sals_age[i_pat,:,:], r_peaks, fs=FS)
    median_signals[i_pat] = {}
    median_signals[i_pat]['median_beats'] = median_beat
    median_signals[i_pat]['median_sals_af'] = median_beat_sal_af
    median_signals[i_pat]['median_sals_sex'] = median_beat_sal_sex
    median_signals[i_pat]['median_sals_age'] = median_beat_sal_age    
    
    if len(median_beat) > max_length:
        max_length = len(median_beat)

np.savez(out_folder+'median_signals_'+name, median_signals = median_signals)

# %% Load median beats subgroups ctrl and af
median_signals    = np.load(out_folder+'median_signals_'+name+'.npz', allow_pickle=True)['median_signals'].item()

#%% Alignment of the ECGs
lengths         = [len(median_signals[i_pat]['median_beats']) 
                   for i_pat in range(len(median_signals))]
median_length   = int(np.median(lengths))

# new points on x axis
new_x_points    = np.linspace(0, 1, median_length)

aligned_beats   = np.zeros((len(median_signals), len(new_x_points), 8))
aligned_sals_af = np.zeros((len(median_signals), len(new_x_points), 8))
aligned_sals_sex = np.zeros((len(median_signals), len(new_x_points), 8))
aligned_sals_age = np.zeros((len(median_signals), len(new_x_points), 8))

# Indices
for i_pat in range(len(median_signals)):
    beat        = median_signals[i_pat]['median_beats']
    beat_length = len(beat)
    
    x_points    = np.linspace(0, 1, beat_length)
    
    # for each point in new_xpoint, find closest in x_points
    indices = [np.argmin(np.abs(x_points - new_x)) for new_x in new_x_points]
    aligned_beats[i_pat, :, :] = beat[indices, :]
    
    sal_af      = median_signals[i_pat]['median_sals_af']
    sal_sex     = median_signals[i_pat]['median_sals_sex']
    sal_age     = median_signals[i_pat]['median_sals_age']
    aligned_sals_af[i_pat, :, :]    = sal_af[indices, :]
    aligned_sals_sex[i_pat, :, :]   = sal_sex[indices, :]
    aligned_sals_age[i_pat, :, :]   = sal_age[indices, :]

np.savez(out_folder+'aligned_median_signals_'+name, median_beats = aligned_beats, 
          median_sals_af = aligned_sals_af, median_sals_sex = aligned_sals_sex,
          median_sals_age = aligned_sals_age)

#%% Load interpolated median beats subgroups ctrl and af
median_beats    = np.load(out_folder+'aligned_median_signals_'+name+'.npz', allow_pickle=True)['median_beats']/1000 # (mV)
median_sals_af  = np.load(out_folder+'aligned_median_signals_'+name+'.npz', allow_pickle=True)['median_sals_af']
median_sals_sex = np.load(out_folder+'aligned_median_signals_'+name+'.npz', allow_pickle=True)['median_sals_sex']
median_sals_age = np.load(out_folder+'aligned_median_signals_'+name+'.npz', allow_pickle=True)['median_sals_age']

#%% AF classication model Saliency Map
i_ctrl      = np.where(y_test == 0)[0]
i_af        = np.where(y_test == 1)[0]
i_female    = np.where(df['Gender'] == 'FEMALE')[0]
i_male      = np.where(df["Gender"] == "MALE")[0]
i_18        = np.where(df['age_bin'] == '[17, 60)')[0]
i_60        = np.where(df["age_bin"] == '[60, 70)')[0]
i_70        = np.where(df['age_bin'] == '[70, 80)')[0]
i_80        = np.where(df["age_bin"] == '[80, 130)')[0]

median_beats_norm   = normalize_max(median_beats)
median_sals_af_norm = percentile_normalize(median_sals_af, low=1, high=99)

vmin = 0; vmax = 1

groups_ecg_af = {'Control': median_beats_norm[i_ctrl, :, :], 
              'AF': median_beats_norm[i_af, :, :]}
groups_ecg_sex = {'Female': median_beats_norm[i_female, :, :], 
              'Male': median_beats_norm[i_male, :, :]}
groups_ecg_age = {'18-60': median_beats_norm[i_18, :, :], 
              '≥80': median_beats_norm[i_80, :, :]}

groups_sal_af_af = {'Control': median_sals_af_norm[i_ctrl, :, :], 
              'AF': median_sals_af_norm[i_af, :, :]}
groups_sal_af_sex = {'Female': median_sals_af_norm[i_female, :, :], 
              'Male': median_sals_af_norm[i_male, :, :]}
groups_sal_af_age = {'18-60': median_sals_af_norm[i_18, :, :], 
              '≥80': median_sals_af_norm[i_80, :, :]}

groups_ecg = [groups_ecg_af, groups_ecg_sex, groups_ecg_age]
groups_sal = [[groups_sal_af_af, groups_sal_af_sex, groups_sal_af_age]]


plot_af_sal(groups_ecg, groups_sal, LEAD_II_IDX, vmin, vmax)

#%% Sex classification
median_sals_sex_norm = percentile_normalize(median_sals_sex)
vmin = 0; vmax = 1

i_ctrl_female    = np.where((y_test==0) & (df['Gender'] == 'FEMALE'))[0]
i_ctrl_male      = np.where((y_test==0) & (df['Gender'] == 'MALE'))[0]

ecg_female       = median_beats_norm[i_ctrl_female, :, :]
ecg_male         = median_beats_norm[i_ctrl_male, :, :]

sal_female       = median_sals_sex_norm[i_ctrl_female, :, :]
sal_male         = median_sals_sex_norm[i_ctrl_male, :, :]

plot_model_sal(ecg_female, ecg_male, sal_female, sal_male, LEAD_II_IDX, 
               'Sex classification', 'Female', 'Male', vmin, vmax)

#%% Age regression
median_sals_age_norm = percentile_normalize(median_sals_age)

i_ctrl_18    = np.where((y_test==0) & (df['age_bin'] == '[17, 60)'))[0]
i_ctrl_80    = np.where((y_test==0) & (df["age_bin"] == '[80, 130)'))[0]

ecg_18 = median_beats_norm[i_ctrl_18, :, :]
ecg_80 = median_beats_norm[i_ctrl_80, :, :]

sal_18 = median_sals_age_norm[i_ctrl_18, :, :]
sal_80 = median_sals_age_norm[i_ctrl_80, :, :]

plot_model_sal(ecg_18, ecg_80, sal_18, sal_80, LEAD_II_IDX, 
               'Age regression', '18-60', '≥80', vmin, vmax)

