###############################################################################
#
# Select control group 
#
# Three criteria:
# 1. Patients has only ECGs with DiagnoseStatement 'Sinusritme' and not atriumfibrilleren and not atriumflutter in diagnose
#
###############################################################################

import pandas as pd


def is_sinus_statement(diagnosis: str) -> bool:
    
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

    return has_sinus and not has_af


def select_control_group(df):
    # Group per PIN and check if a PIN has only 'sinusritme'
    pins_all_sinus = (
        df.groupby("PIN")["DiagnosisFinal"]
          .apply(lambda x: all(is_sinus_statement(diag) for diag in x))
    )
    # Select the PINs satisfying pins_all_sinus
    pins_all_sinus = pins_all_sinus[pins_all_sinus].index
    # Filter originele dataframe
    df = df[df["PIN"].isin(pins_all_sinus)].copy()
    
    print("Number of individuals", len(df["PIN"].unique()))
    print("Number of ECGs", len(df))

    return df


if __name__=='__main__':
	# Main folder
    main_folder     	= "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    df                  = pd.read_excel(main_folder+"preselected_group.xlsx")
    print("Preselected group loaded")
    df_control_group 	= select_control_group(df)
    df_control_group.to_excel(main_folder+"control_group.xlsx", index=False)
    print("Control group created")