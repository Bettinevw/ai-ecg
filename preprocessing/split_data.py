###############################################################################
#
# Stratified split of data sets to train and test data sets
#
###############################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset(file_control, file_af, test_size=0.2, val_size=0.125, random_state=42):
    """
    Split on patient level (PIN) so that each patient appears in exactly one of train/val/test.
    Keeps stratification by Group at the PIN level (based on one Group label per PIN).
    
    Default sizes: test=20%, val=10%, train=70%  (because val_size=0.125 of the remaining 80%)
    """
    # Load
    df_control = pd.read_excel(file_control)
    df_control = df_control[df_control["days_to_index"] != 0] # Exclude index ECG
    df_af      = pd.read_excel(file_af)
    df_af      = df_af[df_af["days_to_index"] != 0] # Exclude index ECG
       
    print("Control Group")
    print("Number of individuals", len(df_control["PIN"].unique()))
    print("Number of ECGs", len(df_control))
    
    print("AF group")
    print("Number of individuals", len(df_af["PIN"].unique()))
    print("Number of ECGs", len(df_af))
    
    # Combine
    df = pd.concat([df_control, df_af], ignore_index=True)
    
    print("Total")
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))

    # Safety checks
    required_cols = {"PIN", "Group", "Gender", "age_at_last_ecg"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    age_bins = np.array([17, 60, 70, 80, 130])
    df["age_bin"] = pd.cut(
        df["age_at_last_ecg"],
        bins=list(age_bins),
        right=False,
        include_lowest=True
    ).astype(str)
    
    
    # Ensure each PIN has exactly one Group (otherwise stratify is ambiguous)
    for col in ["Group", "Gender", "age_bin"]:
        pin_counts = df.groupby("PIN")[col].nunique()
        bad_pins = pin_counts[pin_counts > 1].index.tolist()
        if bad_pins:
            raise ValueError(
                f"Some PINs have multiple '{col}' labels. Fix this first. "
                f"Example PINs: {bad_pins[:10]}"
            )

    # Create PIN-level table for splitting
    df_pins = (
        df[["PIN", "Group", "Gender", "age_bin"]]
        .drop_duplicates(subset=["PIN"])
        .reset_index(drop=True)
    )
    
    df_pins["strata"] = (
        df_pins["Group"].astype(str) + " | " +
        df_pins["Gender"].astype(str) + " | " +
        df_pins["age_bin"].astype(str)
    )
    
    print(df_pins["strata"])
    
    # Split PINs: train+val vs test
    pins_temp, pins_test = train_test_split(
        df_pins["PIN"].values,
        test_size=test_size,
        stratify=df_pins["Group"].values,
        random_state=random_state
    )

    # For val split we need stratify labels for the remaining PINs
    df_pins_temp = df_pins[df_pins["PIN"].isin(pins_temp)].copy()
    
    pins_train, pins_val = train_test_split(
        df_pins_temp["PIN"].values,
        test_size=val_size,
        stratify=df_pins_temp["Group"].values,
        random_state=random_state
    )

    # Map back to full dataframe
    df_train = df[df["PIN"].isin(pins_train)].reset_index(drop=True)
    df_val   = df[df["PIN"].isin(pins_val)].reset_index(drop=True)
    df_test  = df[df["PIN"].isin(pins_test)].reset_index(drop=True)

    return df_train, df_test, df_val


def keep_one_random_ecg_per_individu(df):
    
    df_test_temp = df.copy()

    # Shuffle deterministically
    df_test_temp = df_test_temp.sample(frac=1, random_state=42)

    # Keep first after shuffle → random per patient
    df_test_one_ecg = df_test_temp.drop_duplicates(subset=["PIN"], keep="first")
    
    return df_test_one_ecg
    

    
if __name__ == "__main__": 
    # Configuration
    folder          = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    file_control    = folder + "prepared_control_group.xlsx"
    file_af         = folder + "prepared_AF_group.xlsx"
    
    df_train, df_test_temp, df_val = split_dataset(file_control, file_af)
    
    # Keep only one ecg in de test_set
    df_test = keep_one_random_ecg_per_individu(df_test_temp)


    df_train.to_excel(folder + "dataset_train_stratified.xlsx", index=False)
    df_test.to_excel(folder + "dataset_test_stratified.xlsx", index=False)
    df_val.to_excel(folder + "dataset_val_stratified.xlsx", index=False)



