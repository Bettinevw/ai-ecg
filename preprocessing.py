###############################################################################
#
# Preprocessing MUSE ECG data pipeline
# 1. Couple ECGs to patient
#    The MUSE directory consists of xml files, which contain 1 ECG of 8 leads. 
#    A patient could have multiple ECGs, so the first step is to couple all the
#    ECGs of one patient. Output file contains dictionary:
#    {'PIN': {'date_time': {'path': }}}
# 2. Metadata consists everything of the xml files, excepts for the ECG signal
#    itself. It is saved with in a .parquet   
# 3. Select control group
#    Creates a metadata file (.xlsx) with only patients meeting control group 
#    criteria
# 4. Select AF group
#    Creates a metadata file (.xlsx) with only patients meeting AF group 
#    criteria
# 5. Select ECGs from control group
#    Creates a metadata file (.xlsx) with only ECGs used for 
#    train/validate/test
# 6. Select ECGs from AF group
#    Creates a metadata file (.xlsx) with only ECGs used for 
#    train/validate/test
###############################################################################

import sys
import os 
import pandas as pd

# Add preprocessing directory to work directory
dir_path        = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/preprocessing/')

from couple_ECGs_to_patient import save_file_paths #, test_patient_file
from create_metadata import create_metadata
from preselection import preselection
from select_control_group import select_control_group
from select_af_group import select_af_group
from prepare_ctrl import prepare_ctrl
from prepare_af import prepare_af
from split_data import split_dataset, keep_one_random_ecg_per_individu
from save_signals import save_signal, save_outcome

# ------------------- Folders ----------------------------------------------- #
muse_data_dir   = "G:/diva1/Research/MUSE ECGs/"
xml_folders     = [muse_data_dir+"00_01_02_03/", 
                   muse_data_dir+"04_05_06_07/",
                   muse_data_dir+"12_13_14/",
                   muse_data_dir+"15_16_17/",
                   muse_data_dir+"2018/",
                   muse_data_dir+"tm_1999/"]

name = 'stratified'

metadata_folder                     = muse_data_dir+"PROJECTS/Bettine/Data/"
pkl_file                            = metadata_folder + 'patient_file_paths.pkl'
metadata_file                       = metadata_folder + "metadata.parquet"
preselected_file                    = metadata_folder + "preselected_group.xlsx"
control_group_file                  = metadata_folder + "control_group.xlsx"
af_group_file                       = metadata_folder + "AF_group.xlsx"
prepared_ctrl_file                  = metadata_folder + "prepared_control_group.xlsx"
prepared_af_file                    = metadata_folder + "prepared_AF_group.xlsx"

dataset_training_file               = metadata_folder + "dataset_train_" + name + ".xlsx"
dataset_test_file                   = metadata_folder + "dataset_test_" + name + ".xlsx"
dataset_validation_file             = metadata_folder + "dataset_val_" + name + ".xlsx"
ecg_test_file                       = metadata_folder + "test_dataset_" + name
outcome_test_file                   = metadata_folder + "test_outcome_" + name 
ecg_train_file                      = metadata_folder + "train_dataset_" + name
outcome_train_file                  = metadata_folder + "train_outcome_" + name
ecg_val_file                        = metadata_folder + "val_dataset_" + name
outcome_val_file                    = metadata_folder + "val_outcome_" + name 

# ------------------- Booleans ---------------------------------------------- #
run_couple_ECGs_to_patient      = False
run_create_metadata             = False
run_preselection                = False
run_select_controlgroup         = False
run_select_AFgroup              = False
run_prepare_ctrl                = False
run_prepare_af                  = False
run_split_dataset               = True
run_save_signals                = True

# ------------------- 1. Couple ECGs to patients ---------------------------- #
if run_couple_ECGs_to_patient:
    print('1. Couple ECGs to patients')
    patient_file_paths = save_file_paths(xml_folders, pkl_file)    
# In postprocessing folder, there is a script called plot_ECGs_from_xml.py. Use
# this script to visually check randomly ECGs.
    
# ------------------- 2. Create metadata ------------------------------------ #
if run_create_metadata:
    print('2. Create metadata')
    metadata       = create_metadata(pkl_file)    
    df             = pd.DataFrame(metadata)
    df.to_parquet(metadata_file, index=False)   

# ------------------- 3. Preselection --------------------------------------- # 
if run_preselection:
    # ECG level
    # 0. Create DiagnosisFinal
    # 1. Remove ECGs without diagnosis statement
    # 2. Multiple ECGs at one day, keep only first
    #
    # Individual level
    # 3. Remove all individuals with only 1 ECG
    # 4. Individual age <18 at first ECG recording
    print('3. Preselection')
    df             = pd.read_parquet(metadata_file)
    metadata       = preselection(df)    
    df_preselected = pd.DataFrame(metadata)
    df_preselected.to_excel(preselected_file, index=False)  

# ------------------- 4. Select control group ------------------------------- #    
if run_select_controlgroup:
    # One criteria:
    # 1. Patients has only ECGs with DiagnoseStatement 'Sinusritme'
    print('4. Select control group')
    df_preselected     = pd.read_excel(preselected_file)
    df_control_group 	= select_control_group(df_preselected)
    df_control_group.to_excel(control_group_file, index=False)

# ------------------- 5. Select control group ------------------------------- #     
if run_select_AFgroup:
    # Two criteria:
    # 1: atriumfibrilleren or atriumflutter should be in diagnose statement
    # 2: If 'is veranderd in' is in de statement than 'atriumflutter' or 
    #    'atriumfibrilleren' should be before 'is veranderd in'.
    print('5. Select AF group')
    df_preselected      = pd.read_excel(preselected_file)
    df_AF_group 	    = select_af_group(df_preselected)
    df_AF_group.to_excel(af_group_file, index=False)
    
    
# In postprocessing folder, there is a script called check_control_group.py. 
# Use this script to check randomly if all ECGs of selected patients have 
# 'sinusritme' statement.
max_days_to_index = 3*365.25 # five years
if run_prepare_ctrl:
    # 1. Keep only neccessary columns
    # 3. Add group column and days_to_index
    # 4. PatientAge of patient first ECG
    print('6. prepare ctrl group')
    usecols = ["PIN", "PatientAge", "AgeUnits","DateofBirth", "Gender", "AcquisitionTime",
               "AcquisitionDate", "FilePath", "DiagnosisFinal"]    
    df_ctrl             = prepare_ctrl(control_group_file, max_days_to_index, usecols)
    df_ctrl.to_excel(prepared_ctrl_file, index=False) 

if run_prepare_af:
    # 1. Keep only necessary columns
    # 4. Remove all ECGs with no diagnosis statement
    # 5. Select ECGs and add days_to_index
    # 6. Add group column
    # 7. PatientAge of patient first ECG
    print('7. prepare af group')
    usecols = ["PIN", "PatientAge", "AgeUnits","DateofBirth", "Gender", "AcquisitionTime",
               "AcquisitionDate", "FilePath", "DiagnosisFinal", 'is_afib']  
    df_af          = prepare_af(af_group_file, max_days_to_index, usecols)
    df_af.to_excel(prepared_af_file, index=False)

# ------------------- 8. Split data into test/train group ------------------- # 
if run_split_dataset:
    print('8. Split dataset')
    # Method 1: Only balanced on group
    df_train, df_test_temp, df_val = split_dataset(prepared_ctrl_file, 
                              prepared_af_file, test_size=0.2)
    
    # Keep only one ecg in de test_set
    df_test = keep_one_random_ecg_per_individu(df_test_temp)
    
    df_train.to_excel(dataset_training_file, index=False)
    df_test.to_excel(dataset_test_file, index=False)
    df_val.to_excel(dataset_validation_file, index=False)
    
    print("Training data subset")
    print("Number of individuals", len(df_train["PIN"].unique()))
    print("Number of ECGs", len(df_train))
    
    print("Validation data subset")
    print("Number of individuals", len(df_val["PIN"].unique()))
    print("Number of ECGs", len(df_val))
    
    print("Test data subset")
    print("Number of individuals", len(df_test["PIN"].unique()))
    print("Number of ECGs", len(df_test))
    

# ------------------- 8. Collect all ECG signals in one matrix -------------- #
if run_save_signals:
    print('8. Save signals')
    save_signal(dataset_test_file, ecg_test_file)
    print('Test file')
    save_outcome(dataset_test_file, outcome_test_file)
    print('Test outcome file')
    save_signal(dataset_training_file, ecg_train_file)
    print('Training file')
    save_outcome(dataset_training_file, outcome_train_file)
    print('Training outcome file')
    save_signal(dataset_validation_file, ecg_val_file)
    print('validation file')
    save_outcome(dataset_validation_file, outcome_val_file)
    print('validation outcome file')