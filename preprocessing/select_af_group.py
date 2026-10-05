###############################################################################
#
# Selectie AF group 
#
# TWo criteria:
# 1: atriumfibrilleren or atriumflutter should be in diagnose statement
# 2: If 'is veranderd in' is in de statement than 'atriumflutter' or 
#    'atriumfibrilleren' should be before 'is veranderd in'.


###############################################################################


import pandas as pd

def is_true_afib(diagnosis: str) -> bool:
    
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split("is veranderd in")[0]
    
    has_af = (
        'atriumfibrilleren' in diagnosis or
        'atrial fibrillation' in diagnosis or
        'atriumflutter' in diagnosis or
        'atrial flutter' in diagnosis
    )    

    return has_af


def select_af_group(df):
    # Apply afib rule
    boolean_af    = df['DiagnosisFinal'].fillna('').apply(is_true_afib)
    df['is_afib'] = boolean_af
    pins_af       = df.loc[boolean_af, 'PIN'].unique() 
    df_af 	     = df[df['PIN'].isin(pins_af)].copy() 
    
    print("Number of individuals", len(df_af["PIN"].unique()))
    print("Number of ECGs", len(df_af))

    return df_af


if __name__=='__main__':

	# Main folder
    main_folder     	= "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
    df                  = pd.read_excel(main_folder+"preselected_group.xlsx")
    print("Preselected group loaded")
    df_af_group 		= select_af_group(df)
    df_af_group.to_excel(main_folder + "AF_group.xlsx", index=False)
    print("AF group file saved")