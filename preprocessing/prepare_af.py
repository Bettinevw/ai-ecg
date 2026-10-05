###############################################################################
#
# Prepare AF group 
#
# 1. Keep only neccessary columns
# 4. Select ECGs and add days_to_index
# 5. Add group column
# 6. PatientAge of patient first ECG
#
###############################################################################

import pandas as pd
import numpy as np

from select_control_group import is_sinus_statement


def has_sinus_or_af(diagnosis: str) -> bool:
    
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split("is veranderd in")[0]
    
    has_sinus = (
    'sinusritme' in diagnosis or
    'sinus ritme' in diagnosis or
    'sinus rhythm' in diagnosis
    )
    
    has_af = (
        'atriumfibrilleren' in diagnosis or
        'atrial fibrillation' in diagnosis or
        'atriumflutter' in diagnosis or
        'atrial flutter' in diagnosis
    )    

    return has_sinus or has_af


def select_ecgs(df):
    # Select all ECGs before first AF ECG (index ECG) and first AF ECG

    df['AcquisitionDate'] = pd.to_datetime(df['AcquisitionDate'])
    
    selected_ecgs_with_af = []
    
    for pin, df_pin in df.groupby("PIN"):
        # print(pin)
        # Sort ECGs on date
        df_sorted = df_pin.sort_values("AcquisitionDate")
        
        af_rows     = df_sorted[df_sorted['is_afib'] == True].copy()
        sinus_rows  = df_sorted[df_sorted['is_afib'] == False].copy()
        
        first_af        = af_rows.iloc[0] # first af
        date_first_af   = first_af["AcquisitionDate"]
        
        ecg_sinus   = None
        ecg_af      = None
        
        # Add days to index
        for j in range(len(sinus_rows)):
            row_sinus   = sinus_rows.iloc[j]
            date_sinus  = row_sinus["AcquisitionDate"]
            
            delta_days  = (date_first_af - date_sinus).days

            if delta_days > 0:
                ecg_af      = first_af.copy()  # Save the last
                ecg_sinus   = row_sinus.copy()
                
                ecg_af['days_to_index']       = 0
                ecg_sinus['days_to_index']    = delta_days

            else: # delta_days <= 0: sinusrhythm is after AF
                continue

            if ecg_sinus is not None:
                selected_ecgs_with_af.append(ecg_sinus)
        
        if ecg_sinus is not None:
            selected_ecgs_with_af.append(ecg_af)
    
    # Make DataFrame with selected ECGs
    df_selected_with_af   = pd.DataFrame(selected_ecgs_with_af)

    return df_selected_with_af


def prepare_af(AF_group_file, max_days_to_index, usecols):
    # 1. keep only neccessary columns  
    df      = pd.read_excel(AF_group_file, usecols = usecols)

    # 4. Select all ECGs in sinusritme or af
    boolean_af    = df['DiagnosisFinal'].fillna('').apply(has_sinus_or_af)
    pins_af       = df.loc[boolean_af, 'PIN'].unique() 
    df 	     = df[df['PIN'].isin(pins_af)].copy() 
    
    # 5. Select ECGs and add days_to_index
    df = select_ecgs(df)
    
    # 6. Add needed columns for analysis
    df["Group"] = "AF"

    # Add PatientAge at first and last ecg
    first_age = df.groupby("PIN", sort=False)["PatientAge"].transform("min")
    df["age_at_first_ecg"] = first_age
    last_age = df.groupby("PIN", sort=False)["PatientAge"].transform("max")
    df["age_at_last_ecg"] = last_age
    
    # Exclude ECGs further away from index ECG than max_day_to_index
    df = df[df['days_to_index'] < max_days_to_index]
    
    # Remove all patients with 1 ECG
    df = df[df.groupby("PIN")["PIN"].transform('count') > 1]
    
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    
    return df


if __name__=='__main__':
	# Main folder
    main_folder     	= "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    AF_group_file  = main_folder + "AF_group.xlsx"
    
    usecols = ["PIN", "PatientAge", "AgeUnits","DateofBirth", "Gender", "AcquisitionTime",
               "AcquisitionDate", "FilePath", "DiagnosisFinal", 'is_afib']  
    
    max_days_to_index = 3*365.25 # 5 years
    df_af 	= prepare_af(AF_group_file, max_days_to_index, usecols)
    
    df_af.to_excel(main_folder + "prepared_AF_group.xlsx", index=False)