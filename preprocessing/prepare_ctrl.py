###############################################################################
#
# Prepare control group 
#
# 1. Keep only neccessary columns
# 3. Add group column and days_to_index
# 4. PatientAge of patient last ECG
#
###############################################################################

import pandas as pd
import numpy as np


def mark_index_ecg(df, time_col="AcquisitionDate", id_col="PIN"):
    # Days to index ECG
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    last_dates = df.groupby(id_col, sort=False)[time_col].transform("max")

    df["days_to_index"] = (last_dates - df[time_col]).dt.days
    
    return df    



def prepare_ctrl(control_group_file, max_days_to_index, usecols):
    # 1. keep only neccessary columns
    df      = pd.read_excel(control_group_file, usecols = usecols)

    # 4. Add needed columns for analysis
    df["Group"] = "Control"
    df = mark_index_ecg(df)
    
    # Add PatientAge at first and last ecg
    first_age = df.groupby("PIN", sort=False)["PatientAge"].transform("min")
    df["age_at_first_ecg"] = first_age
    last_age = df.groupby("PIN", sort=False)["PatientAge"].transform("max")
    df["age_at_last_ecg"] = last_age
    
    # Exclude ECGs further away from index ECG than max_day_to_index
    df = df[df['days_to_index'] < max_days_to_index]
    
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    
    # Remove individuals with only one ECG
    counts = df['PIN'].value_counts()
    patients_with_multiple_ecgs = counts[counts >= 2].index
    df  = df[df['PIN'].isin(patients_with_multiple_ecgs)].copy()
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))

    return df
    

if __name__=='__main__':
	# Main folder
    main_folder     	= "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    control_group_file  = main_folder + "control_group.xlsx"
    
    usecols = ["PIN", "PatientAge", "AgeUnits","DateofBirth", "Gender", "AcquisitionTime",
               "AcquisitionDate", "FilePath", "DiagnosisFinal"]   
    
    max_days_to_index = 3*365.25
    
    df_ctrl 	= prepare_ctrl(control_group_file, max_days_to_index, usecols)
    
    df_ctrl.to_excel(main_folder + "prepared_control_group.xlsx", index=False)
