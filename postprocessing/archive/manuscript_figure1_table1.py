import pickle
import pandas as pd
import numpy as np
import scipy
from scipy.stats import mannwhitneyu, chi2_contingency

# Folders
muse_data_dir = "G:/diva1/Research/MUSE ECGs/"
metadata_folder = muse_data_dir+"PROJECTS/Bettine/Data/"
manuscript_folder = muse_data_dir+"PROJECTS/Bettine/Manuscript/Revision/"

# Files
pkl_file = metadata_folder + 'patient_file_paths.pkl'   
prepared_ctrl_file = metadata_folder + "prepared_control_group.xlsx"
prepared_af_file = metadata_folder + "prepared_AF_group.xlsx"
name_out = 'stratified'
dataset_training_file = metadata_folder + "dataset_train_" + name_out + ".xlsx"
dataset_unbalanced_test_file = metadata_folder + "dataset_test_stratified.xlsx"
dataset_balanced_test_file = metadata_folder + "dataset_test_prevalence.xlsx"
dataset_validation_file = metadata_folder + "dataset_val_" + name_out + ".xlsx"

df_ctrl    = pd.read_excel(prepared_ctrl_file)
df_af      = pd.read_excel(prepared_af_file)
df_train   = pd.read_excel(dataset_training_file)
df_test_balanced = pd.read_excel(dataset_balanced_test_file)
df_test_unbalanced = pd.read_excel(dataset_unbalanced_test_file)
df_val     = pd.read_excel(dataset_validation_file)

print('Total % AF:', len(df_af['PIN'].unique())/(len(df_ctrl['PIN'].unique()) + len(df_af['PIN'].unique()))*100)

df_ctrl = df_ctrl[df_ctrl["days_to_index"] != 0] # Exclude index ECG
df_af = df_af[df_af["days_to_index"] != 0] # Exclude index ECG

print('Total % AF:', len(df_af['PIN'].unique())/(len(df_ctrl['PIN'].unique()) + len(df_af['PIN'].unique()))*100)


table = {'Nr of individuals': [],
             'Sex (% female)': [], 
             'Develops AF (%)': [],
             'Age at first recorded ECG years (mean ± std)': [],
             'Mean age years (mean ± std)': [],
             'Age at index ECG years (mean ± std)': [],
             'Time between first recorded and index ECG days (mean ± std)': [], 
             'Number of ECGs per person (mean ± std)': []}

df_all = [df_ctrl, df_af, df_train, df_val, df_test_unbalanced, df_test_balanced]
df_names = ['ctrl', 'AF', 'Train', 'Val', 'Test Unbalanced', 'Test Balanced']

save_ecgs_per_patient = {}

count = 0 
for df in df_all:
    df_unique = df.groupby("PIN", as_index=False).agg({
                                "age_at_first_ecg": "first",
                                "PatientAge": "mean",
                                "age_at_last_ecg": "first",
                                "Gender": "first",
                                "Group": "first",
                                "days_to_index":"max"})
    af_perc_ind = len(df_unique[df_unique['Group']=='AF'])/len(df_unique)*100
    af_perc_ecg = len(df_unique[df_unique['Group']=='AF'])/len(df)*100
    mean_mean_age = np.mean(df_unique["PatientAge"])
    std_mean_age = np.std(df_unique["PatientAge"])
    mean_first_age = np.mean(df_unique["age_at_first_ecg"])
    std_first_age = np.std(df_unique["age_at_first_ecg"])
    mean_age = np.mean(df_unique["age_at_last_ecg"])
    std_age = np.std(df_unique["age_at_last_ecg"])
    female_perc = len(df_unique[df_unique['Gender']=='FEMALE'])/len(df_unique)*100
    mean_days = np.mean(df_unique["days_to_index"])
    std_days = np.std(df_unique["days_to_index"])
    nr_ecgs_per_pat = []
    for pin in df['PIN'].unique():
        pat = df[df['PIN']==pin]
        nr_ecgs_per_pat = np.append(nr_ecgs_per_pat, len(pat))
    mean_ecg_per_pat = np.mean(nr_ecgs_per_pat)
    std_ecg_per_pat = np.std(nr_ecgs_per_pat)
    
    save_ecgs_per_patient[df_names[count]]= nr_ecgs_per_pat
    
    table['Nr of individuals'].append(len(df_unique))
    table['Sex (% female)'].append(np.round(female_perc))
    table['Develops AF (%)'].append(np.round(af_perc_ind))
    table['Age at first recorded ECG years (mean ± std)'].append(str(np.round(mean_first_age,1))+'±'+str(np.round(std_first_age,1)))
    table['Mean age years (mean ± std)'].append(str(np.round(mean_mean_age,1))+'±'+str(np.round(std_mean_age,1)))
    table['Age at index ECG years (mean ± std)'].append(str(np.round(mean_age,1))+'±'+str(np.round(std_age,1)))
    table['Time between first recorded and index ECG days (mean ± std)'].append(str(np.round(mean_days,0))+'±'+str(np.round(std_days,0)))
    table['Number of ECGs per person (mean ± std)'].append(str(np.round(mean_ecg_per_pat,1))+'±'+str(np.round(std_ecg_per_pat,1)))
    
    count += 1

#%%
print('Total individual:', len(df_ctrl['PIN'].unique()) + len(df_af['PIN'].unique()))
print('Total ECG recordings:', len(df_ctrl['PIN']) + len(df_af['PIN']))
print('Total % AF:', len(df_af['PIN'].unique())/(len(df_ctrl['PIN'].unique()) + len(df_af['PIN'].unique()))*100)

# Sex distribution
df_ctrl_uni = df_ctrl.groupby("PIN", as_index=False).agg({
                            "age_at_last_ecg": "first",
                            "Gender": "first",
                            "Group": "first",
                            "days_to_index":"max"})
df_af_uni = df_af.groupby("PIN", as_index=False).agg({
                            "age_at_first_ecg": "first",
                            "PatientAge": "mean",
                            "age_at_last_ecg": "first",
                            "Gender": "first",
                            "Group": "first",
                            "days_to_index":"max"})

control_counts = df_ctrl_uni['Gender'].value_counts().reindex(['FEMALE', 'MALE'], fill_value=0)
af_counts = df_af_uni['Gender'].value_counts().reindex(['FEMALE', 'MALE'], fill_value=0)

# Maak de contingency table
contingency_table = pd.DataFrame({
    'Control': control_counts,
    'AF': af_counts
})

# Chi square
chi2, p, dof, expected = chi2_contingency(contingency_table)
print('p_sex: ', p)
# Continuous variables were compared using the Mann–Whitney U test. 
stat_age, p_age = mannwhitneyu(df_ctrl_uni["age_at_last_ecg"], df_af_uni["age_at_last_ecg"])
print('p_age: ', p_age)
stat_interval, p_interval = mannwhitneyu(df_ctrl_uni["days_to_index"], df_af_uni["days_to_index"])
print('p_interval: ', p_interval)
stat_ecgs, p_ecgs = mannwhitneyu(save_ecgs_per_patient["ctrl"], save_ecgs_per_patient["AF"])
print('p_ecgs: ',p_ecgs)

# Column names
columns = ['Control group', 'AF group', 'Training', 'Validation', 'Unbalanced Test', 'Balanced Test']

# Create dataframe
df_table = pd.DataFrame(table, index=columns)

# Transpose such that the grups are columns
df_table = df_table.T

df_table.to_excel(manuscript_folder+'table1.xlsx')

