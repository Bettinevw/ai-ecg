import xmltodict
import base64
import numpy as np
import pandas as pd
from copy import copy
import h5py


def _decode_wave(lead_xml):
    
    LeadAmplitudeUnitsPerBit = float(lead_xml['LeadAmplitudeUnitsPerBit'])
    
    # Read the b64 data
    b64_encoded = ''.join(lead_xml['WaveFormData'].split('\n'))
    # Decode it from b64
    decoded = base64.b64decode(b64_encoded)
    # Convert decoded bytes to int
    signal = np.frombuffer(decoded, dtype='int16')
    # Fix amplitude    
    signal = signal*LeadAmplitudeUnitsPerBit
    
    return signal


def save_signal(data_file, npy_out):
    
    df      = pd.read_excel(data_file)
    
    # Writes directly on disk instead using RAM
    ecg = np.lib.format.open_memmap(
        npy_out + ".npy", mode="w+",
        dtype=[('ecg', 'float32', (2500, 8)), ('PIN', 'S9')],
        shape=(len(df),)
    )
    
    PINs    = df['PIN'].astype(str).to_numpy()
    PINs    = np.char.zfill(PINs.astype(str), 9)
    
    count_row = 0
    for _, row in df.iterrows():
        # Add PIN
        ecg['PIN'][count_row] = str(row['PIN']).zfill(9).encode('ascii')
        
        # Load xml file
        with open(row['FilePath'], "rb") as fd:
            xml_dict = xmltodict.parse(fd.read(), process_namespaces=True)
        
        mean_wave, leads = xml_dict['RestingECG']['Waveform'] 

        for lead in range(len(leads['LeadData'][:])):
            if leads['LeadData'][lead]['LeadID'] in np.array(['I','II','V1','V2',
                                                              'V3','V4','V5','V6']):
                # Decode b64 data
                wave = _decode_wave(leads['LeadData'][:][lead])
                
                # If length of signal is 5000.
                if len(wave) == 5000:
                    segment     = wave[::2] # Select every two datapoints
                elif len(wave) == 2500:
                    segment     = copy(wave)
                else:
                    print('error: signal length is not 2500 or 5000')

                ecg['ecg'][count_row, :, lead] = segment
        count_row += 1
    
    np.savez(npy_out, ecg=ecg, PIN=PINs)
    
    # return
    
        
def save_outcome(data_file, npy_out):
    
    df = pd.read_excel(data_file)
    
    # PIN
    PINs = df['PIN'].astype(str).to_numpy()
    PINs = np.char.zfill(PINs.astype(str), 9)
    # Outcome
    outcome_str = df['Group']
    outcome_b = outcome_str.replace('Control', 0)
    outcome = outcome_b.replace('AF', 1)
    outcome = outcome.to_numpy()
    # Gender
    out_sex_temp1 = df['Gender']
    out_sex_temp2 = out_sex_temp1.replace('MALE', 0)
    out_sex = out_sex_temp2.replace('FEMALE', 1)
    # Age
    out_age = df['age_at_last_ecg'].astype("float32").to_numpy()
    age_bin = df['age_bin']
    # Time
    days_per_year = 365.25
    time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 
                           1, 2, 3])*days_per_year
    time_labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
    df["time_bin"] = pd.cut(df['days_to_index'], bins=time_bins, 
                             labels=time_labels, right=False)
    time_bin = df['time_bin']
    
    np.savez(npy_out, outcome=outcome, PIN=PINs, 
             out_sex=out_sex, out_age=out_age,
             age_bin=age_bin, time_bin=time_bin)
    
    # return outcome
    

if __name__ == "__main__":
    folder          = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    
    
    name = 'stratified'
    
    test_file      = folder+ "dataset_test_" + name + ".xlsx"
    npy_ecg        = folder+"test_dataset_" + name
    npy_out        = folder+"test_outcome_" + name 
    
    save_signal(test_file, npy_ecg)
    save_outcome(test_file, npy_out)
    print('Test finished')

    train_file      = folder+"dataset_train_" + name + ".xlsx"        
    npy_out         = folder+"train_dataset_" + name
    csv_out         = folder+"train_outcome_" + name     
    
    save_signal(train_file, npy_out)
    save_outcome(train_file, csv_out)
    print('Train finished')
    
    val_file      = folder+"dataset_val_" + name + ".xlsx"        
    npy_out         = folder+"val_dataset_" + name
    csv_out         = folder+"val_outcome_" + name    
    
    save_signal(val_file, npy_out)
    save_outcome(val_file, csv_out)
    print('Validation finished')
    


