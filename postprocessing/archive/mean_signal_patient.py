import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt



def bandpass_filter(sig, fs, low=5, high=15, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def detect_rpeaks_simple(sig, fs, distance_ms=250, height_factor=0.5):
    """
    Very simple R-peak detector on a single lead.

    sig: 1D array
    fs: sampling frequency
    distance_ms: minimale afstand tussen R-toppen
    height_factor: factor * max(sig) als minimale piekhoogte
    """
    # 1. Filter
    filtered = bandpass_filter(sig, fs, low=5, high=15, order=2)

    # 2. Vierkanten en eventueel gladstrijken zou kunnen, maar we houden het simpel
    # squared = filtered**2

    # 3. Piekdetectie
    distance_samples = int(distance_ms * fs / 1000)
    height_thresh = height_factor * np.max(filtered)

    peaks, _ = find_peaks(filtered, distance=distance_samples, height=height_thresh)

    return peaks


def get_rpeaks(signal, fs=250, lead_idx=0):
    """
    signal: (n_leads, n_samples)
    lead_idx: welke lead te gebruiken voor R-peak detectie
    """
    ref_lead = signal[:, lead_idx]
    r_peaks = detect_rpeaks_simple(ref_lead, fs)
    return r_peaks


def compute_avg_beat(signal, r_peaks, fs=250, pre_ms=200, post_ms=400):
    """
    Compute average beat for one patient.
    
    signal: 1D numpy array, ECG time series
    r_peaks: 1D array of R-peak sample indices
    fs: sampling frequency in Hz
    pre_ms, post_ms: window around R-peak in milliseconds
    """
    n_samples, n_leads = signal.shape
    pre_samples  = int(pre_ms  * fs / 1000)
    post_samples = int(post_ms * fs / 1000)
    beat_len = pre_samples + post_samples

    beats = []
    for r in r_peaks:
        start = r - pre_samples
        end   = r + post_samples
        if start < 0 or end > n_samples:
            continue
        beat = signal[start:end, :]            # shape (beat_len, n_leads)
        if beat.shape[0] == beat_len:
            beats.append(beat)

    if len(beats) == 0:
        return None

    beats = np.stack(beats, axis=1)            # shape (n_beats, beat_len, n_leads)
    avg_beat = beats.mean(axis=1)              # shape (beat_len, n_lead )
    return avg_beat


folder          = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/patient_groups/"
folder_out      = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/patient_groups_signal/"
files           = os.listdir(folder)

for i_file in range(len(files)):
    print(files[i_file])
    
    group_file = files[i_file]
    ecg        = np.load(folder+group_file)["ecg"]
    
    output_path = folder_out + group_file[:-4] +"_avg_beats.npy"

    avg_beats   = []   # list of arrays (n_leads, n_samples)
    all_rpeaks    = []   # list of arrays (n_rpeaks,)
    
    for i_ecg in range(len(ecg[:,0,0])):
        ecg_pat         = ecg[i_ecg,:,:]
    
        r_peaks = get_rpeaks(ecg_pat, fs=250, lead_idx=0)
        avg_beat = compute_avg_beat(ecg_pat, r_peaks, fs=250, pre_ms=200, post_ms=400)
        
        if type(avg_beat)==np.ndarray:        
            avg_beats.append(avg_beat)
            all_rpeaks.append(r_peaks)
    
    avg_beats = np.stack(avg_beats, axis=0)   # (n_patients, n_timepoints, n_leads)
    np.savez(output_path, avg_beats=avg_beats, rpeaks=all_rpeaks)

# fig_n, ax_n = plt.subplots(8, 1)
# # fig_m, ax_m = plt.subplots(8, 1)

# for i in range(8):
#     ax_n[i].plot(ecg_pat[:, i], 'k')
#     ax_n[i].plot(avg_beat[:, i], 'b')
#     ax_n[i].plot([r_peaks, r_peaks], [min(ecg_pat[:, i]), max(ecg_pat[:, i])], 'red')
    
