###############################################################################
#
# Preselection
# 
# ECG level
# 0. Create DiagnosisFinal
#    Fill in missing data and remove inconsistent data
# 1. Remove ECGs without diagnosis statement
# 2. Multiple ECGs at one day, keep only first
#
# Individual level
# 3. Individual age <18 at first ECG recording
# 4. Remove all individuals with only 1 ECG
#
###############################################################################

import pandas as pd
import numpy as np


def fill_missing_data(df, parameter):
    # 3. Fill in empty gender and remove comflicting and missing gender cases
    to_drop = []  
    
    if parameter == "PatientAge":
        PatientAge = (pd.to_datetime(df["AcquisitionDate"], errors="coerce")
                      - pd.to_datetime(df["DateofBirth"], errors="coerce")).dt.days / 365.25
        mask = (df[parameter].isna())
        df.loc[mask, parameter] = np.round(PatientAge[mask]).astype(int)
        
    else:    
        for pin in df["PIN"].unique():
            df_pat  = df[df["PIN"] == pin]
            values = df_pat[parameter]
            
            if not values.isna().any():
                # all genders are filled and the same
                continue
            elif values.dropna().empty:
                # Completely missing - remove case
                to_drop.append(pin)
            else:
                if values.dropna().nunique() == 1:
                   # All non-NaN the same -> fill NaNs for this patient
                   fill_val = values.dropna().iloc[0]
                   mask = (df["PIN"] == pin) & (df[parameter].isna())
                   df.loc[mask, parameter] = fill_val
                else:
                    # Conflict - remove case
                    to_drop.append(pin)
                
    if to_drop:
        df = df[~df["PIN"].isin(to_drop)]
    
    return df


# Helper function to determine priority
def get_priority(diagnosis: str) -> int:
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split("is veranderd in")[0]

    has_af = (
        'atriumfibrilleren' in diagnosis or
        'atrial fibrillation' in diagnosis or
        'atriumflutter' in diagnosis or
        'atrial flutter' in diagnosis
    )
    has_sinus = (
        'sinusritme' in diagnosis or
        'sinus ritme' in diagnosis or
        'sinus rhythm' in diagnosis
    )

    if has_af:
        return 0  # Highest priority
    elif has_sinus:
        return 1  # Second priority
    else:
        return 2  # Lowest priority
     

def dedup_same_day(df, id_col="PIN", date_col="AcquisitionDate", 
                   time_col="AcquisitionTime", diagnosis_col="DiagnosisFinal"):
    """
    Keeps only one ECG per day per patient, prioritizing:
    1. AF (atrial fibrillation/flutter)
    2. Sinus rhythm
    3. The first ECG of the day
    """
    df = df.copy()

    # Convert date and time to datetime/timedelta
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if time_col in df.columns:
        df[time_col] = pd.to_timedelta(df[time_col], errors="coerce")
    else:
        df[time_col] = pd.NaT

    # Sort by priority, then by time
    df['priority'] = df[diagnosis_col].apply(get_priority)
    df = df.sort_values(
        [id_col, date_col, 'priority', time_col],
        na_position="last"
    )

    # Keep only the first row per patient per day
    df = df.drop_duplicates(subset=[id_col, date_col], keep="first")
    df = df.drop(columns=['priority'])

    return df


def preselection(df):
    # Subsets that satisfy one of the criteria mentioned at the top of this script
    # Combine diagnosisStatements and fill empty cells
    df["DiagnosisFinal"] = df["DiagnosisStatement"].fillna(df["OriginalDiagnosisStatement"])
    df["DiagnosisFinal"] = df["DiagnosisFinal"].fillna('')
    N_ind = len(df["PIN"].unique())
    N_ecg = len(df)
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    
    # Fill missing data; delete inconsistent data
    df = fill_missing_data(df, "AgeUnits")
    df = fill_missing_data(df, "DateofBirth")
    df = fill_missing_data(df, "Gender")
    df = fill_missing_data(df, "PatientAge")    
    print("Fill in missing data and delete inconsistent data: N_ind=", N_ind-len(df["PIN"].unique()), 
          "N_ecgs=", N_ecg-len(df))
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    N_ind = len(df["PIN"].unique())
    N_ecg = len(df)
        
    # 1. Remove all ECGs with empty DiagnosisStatements
    df = df[df['DiagnosisFinal'].str.len() > 0].copy()
    print("Removal of empty diagnosisStatements: N_ind=", N_ind-len(df["PIN"].unique()), 
          "N_ecgs=", N_ecg-len(df))
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    N_ind = len(df["PIN"].unique())
    N_ecg = len(df)
    
    # 2. Multiple ECGs in one day, keep AF>SR>first of the day
    df = dedup_same_day(df)
    print("Keep only first ECG of the day: N_ind=", N_ind-len(df["PIN"].unique()), 
          "N_ecgs=", N_ecg-len(df))
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    N_ind = len(df["PIN"].unique())
    N_ecg = len(df)
    
    # All patient age in year units
    df["PatientAge"] = pd.to_numeric(df["PatientAge"], errors="coerce").fillna(0)
    df.loc[df["AgeUnits"] == "MONTHS", "PatientAge"] = df.loc[df["AgeUnits"] == "MONTHS", "PatientAge"]/12
    df.loc[df["AgeUnits"] == "MONTHS", "AgeUnits"] = "YEARS"
    
    # Add PatientAge at first ECG    
    first_age = df.groupby("PIN", sort=False)["PatientAge"].transform("min")
    df["age_at_first_ecg"] = first_age
    
    # 3. Select individuals older than 18 at age of first ECG
    df = df[first_age >= 18].copy()
    print("Remove individuals with age < 18 at first ECG: N_ind=", N_ind-len(df["PIN"].unique()), 
          "N_ecgs=", N_ecg-len(df))
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    N_ind = len(df["PIN"].unique())
    N_ecg = len(df)
    
    # 4. Remove all individuals with only 1 ECG
    counts = df['PIN'].value_counts()
    patients_with_multiple_ecgs = counts[counts >= 2].index
    df  = df[df['PIN'].isin(patients_with_multiple_ecgs)].copy()
    print("Remove individuals with 1 ECG: N_ind=", N_ind-len(df["PIN"].unique()), 
          "N_ecgs=", N_ecg-len(df))
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))
    N_ind = len(df["PIN"].unique())
    N_ecg = len(df)
        
    # Save as string
    df["AcquisitionTime"] = ((pd.Timestamp("1970-01-01") + 
                              ( df["AcquisitionTime"] % pd.Timedelta("1D")))
                             .dt.strftime("%H:%M:%S"))

    return df


if __name__=='__main__':
	# Main folder
    main_folder     	= "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    metadatafile    	= main_folder+"metadata.parquet"
    df              	= pd.read_parquet(metadatafile) 
    print("Metadata file loaded")
    df_preselected 	= preselection(df)
    df_preselected.to_excel(main_folder+"preselected_group.xlsx", index=False)
    print("Preselected group created")