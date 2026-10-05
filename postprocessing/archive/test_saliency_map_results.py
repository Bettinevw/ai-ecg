# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# Test
# 1. R-peaks
# 2. Median signals - per patient
# 3. Interpolated signals - per patient
# 4. Mean signals

#%% Load data
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
out_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Images/"
name = 'stratified'

test_file = folder + "dataset_test_" + name + ".xlsx"
df      = pd.read_excel(test_file)
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")

#%% Saliency maps
sals_af     = np.load(out_folder+'saliency.npz', allow_pickle=True)['saliency_af']

#%% Median signals
median_signals    = np.load(out_folder+'median_signals.npz', allow_pickle=True)['median_signals'].item()

#%% Load interpolated median beats subgroups ctrl and af
median_beats    = np.load(out_folder+'aligned_median_signals.npz', allow_pickle=True)['median_beats'] # (mV)
median_sals_af  = np.load(out_folder+'aligned_median_signals.npz', allow_pickle=True)['median_sals_af']
median_sals_sex = np.load(out_folder+'aligned_median_signals.npz', allow_pickle=True)['median_sals_sex']
median_sals_age = np.load(out_folder+'aligned_median_signals.npz', allow_pickle=True)['median_sals_age']

#%% test saliency maps
ind_pat = 106
ecg_pat = X_test[ind_pat, :, 0]
sal_pat = sals_af[ind_pat, :, 0]
rel_time = np.arange(0, np.shape(ecg_pat)[0])/(np.shape(ecg_pat)[0]-1)
            
# Plot ECG
plt.plot(rel_time, ecg_pat, color='black')
# Plot Saliency
bg = np.tile(sal_pat, (1200, 1))
plt.imshow(
    bg, aspect="auto",
    extent=[0, 1, -200, 1200],
    origin="lower", cmap='Purples')

#plt.set_ylim([-1.5, 1.5])
#plt.set_xlabel('Time [ms]')
#plt.set_ylabel(r"Normalized ECG amplitude [mV]")
#plt.annotate("AF model" , 
 #                     xy=(0, 0.05), xycoords="axes fraction", 
  #   xytext=(-90, 0), textcoords='offset pixels',
   #  fontsize=15, rotation="vertical")


#%% Test R-peaks
i_list_pat = np.random.randint(len(X_test), size=12)

# Maak een 3x4 figuur
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, i_pat in enumerate(i_list_pat):
    # Pak het ECG-signaal van lead 0
    sig = X_test[i_pat, :, 0]

    # Detecteer R-peaks
    signals, info = nk.ecg_peaks(sig, method='kalidas2017', sampling_rate=250, correct_artifacts=False)
    r_peaks = info['ECG_R_Peaks']

    # Plot het signaal en de R-peaks
    axes[idx].plot(sig, label='ECG')
    axes[idx].scatter(r_peaks, sig[r_peaks], color='red', label='R-peaks')
    axes[idx].set_title(f'Patient {i_pat}')
    axes[idx].legend()

plt.tight_layout()
plt.show()
    
#%% Test median signal
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()
for idx, i_pat in enumerate(i_list_pat):
    # Test R_peaks
   
    axes[idx].plot(X_test[i_pat, :, 0], 'black')
    axes[idx].plot(median_signals[i_pat]['median_beats'][:, 0], 'tab:green')
    axes[idx].set_title(f'Patient {i_pat}')
plt.tight_layout()
plt.show()
    
#%% test aignment    
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()
for idx, i_pat in enumerate(i_list_pat):
    x_1 = np.linspace(0,1,len(median_beats[i_pat, :, 0]))
    axes[idx].plot(x_1, median_beats[i_pat, :, 0], 'o', label='aligned')
    x_2 = np.linspace(0,1,len(median_signals[i_pat]['median_beats'][:, 0]))
    axes[idx].plot(x_2, median_signals[i_pat]['median_beats'][:, 0], 'o', label='original')
    axes[idx].legend()
    

    
    
    