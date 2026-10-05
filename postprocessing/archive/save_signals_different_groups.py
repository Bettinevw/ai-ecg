import numpy as np
import pandas as pd

import xmltodict
import base64
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

main_folder   = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
AF_file       = main_folder + "prepared_AF_group.xlsx"
control_file  = main_folder + "prepared_control_group.xlsx"

# 1) Load only needed columns to speed up
usecols_af   = ["PIN","Gender","PatientAge","Group","days_to_index", "age_at_last_ecg", "FilePath"]
usecols_ctrl = ["PIN","Gender","PatientAge","Group","days_to_index", "age_at_last_ecg", "FilePath"]
df_a    = pd.read_excel(AF_file,   usecols=usecols_af)
df_c    = pd.read_excel(control_file, usecols=usecols_ctrl)
print('Af and Ctrl group loaded')

df = pd.concat([df_c, df_a], ignore_index=True)
df = df[df["days_to_index"] > 0].copy() # Exclude index ECG

df = df.dropna()

df['AgeBin'] = pd.cut(df['age_at_last_ecg'], bins=[17, 60, 70, 80, 130], 
                      labels=["[18, 60]", "(60, 70]", "(70, 80]", "80+"])
days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3, 4, 5])*days_per_year

df['TimeBin'] = pd.cut(df['days_to_index'], bins=time_bins, 
                      labels=["1-7d", "7d-1m", "1m-3m", "3m-6m", '6m-1y', "1y-2y", "2y-3y", "3y-4y", "4y-5y"])


df['Strata'] = df['Group'].astype(str) + "_" + df['Gender'].astype(str) + "_" + df['AgeBin'].astype(str)+ "_" + df['TimeBin'].astype(str)


strata_groups = df['Strata'].unique()

for i in range(len(strata_groups)):
    print(strata_groups[i])
    df_strata = df[df['Strata']==strata_groups[i]] 
    npy_out = main_folder + strata_groups[i]
    
    # Writes directly on disk instead using RAM
    ecg = np.lib.format.open_memmap(
        npy_out + ".npy", mode="w+",
        dtype=[('ecg', 'float32', (2500, 8)), ('PIN', 'S9')],
        shape=(len(df_strata),)
    )
    
    count_row = 0
    for _, row in df_strata.iterrows():
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
                
                # If length of signal is 5000, get middle part.
                if len(wave) == 5000:
                    segment     = wave[::2] # Select ►every two datapoints
                elif len(wave) == 2500:
                    segment     = copy(wave)
                else:
                    print('error: signal length is not 2500 or 5000')

                ecg['ecg'][count_row, :, lead] = segment
        count_row += 1