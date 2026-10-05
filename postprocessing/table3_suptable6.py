import pickle
import pandas as pd
import numpy as np
import scipy
from scipy.stats import mannwhitneyu, chi2_contingency

# Folders
folder      = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
manuscript_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Manuscript/Submission2/"
# name = 'stratified'
name = 'prevalence'

dataset_test_file   = folder + "dataset_test_" + name + ".xlsx"
df                  = pd.read_excel(dataset_test_file )

days_per_year   = 365.25
time_bins       = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 
                            1, 2, 3])*days_per_year
labels          = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"]  = pd.cut(df['days_to_index'], bins=time_bins, labels=labels, right=False)
df["time_bin"]  = pd.Categorical(df["time_bin"],
                                 categories=labels[::-1],
                                 ordered=True)

age_bins        = [17, 60, 70, 80, 200]
age_labels      = ["18–60","60-70", "70–80", "≥80"]
df["age_bin"]   = pd.cut(
        df["PatientAge"], bins=age_bins, labels=age_labels,
        right=False, include_lowest=True
)

#%%
df_unique = df.groupby("PIN", as_index=False).agg({
                            "time_bin": "first",
                            "age_bin": "first",
                            "Gender": "first",
                            "Group": "first"})

table = {'Nr of individuals': [],
         'Develops AF (%)': [],
             'Sex (% female)': []}

for gender in ['FEMALE', 'MALE']:    
    subgroup_df = df[df["Gender"] == gender]    
    af_perc_ind = len(subgroup_df[subgroup_df ['Group']=='AF'])/len(subgroup_df)*100
    female_perc = len(subgroup_df [subgroup_df ['Gender']=='FEMALE'])/len(subgroup_df)*100
    
    table['Nr of individuals'].append(len(subgroup_df))
    table['Sex (% female)'].append(np.round(female_perc))
    table['Develops AF (%)'].append(np.round(af_perc_ind))
    
for age in age_labels:    
    subgroup_df = df[df["age_bin"] == age]    
    af_perc_ind = len(subgroup_df[subgroup_df ['Group']=='AF'])/len(subgroup_df)*100
    female_perc = len(subgroup_df [subgroup_df ['Gender']=='FEMALE'])/len(subgroup_df)*100
    
    table['Nr of individuals'].append(len(subgroup_df))
    table['Sex (% female)'].append(np.round(female_perc))
    table['Develops AF (%)'].append(np.round(af_perc_ind))
    

for time in labels[::-1]:    
    subgroup_df = df[df["time_bin"] == time]    
    af_perc_ind = len(subgroup_df[subgroup_df ['Group']=='AF'])/len(subgroup_df)*100
    female_perc = len(subgroup_df [subgroup_df ['Gender']=='FEMALE'])/len(subgroup_df)*100
    
    table['Nr of individuals'].append(len(subgroup_df))
    table['Sex (% female)'].append(np.round(female_perc))
    table['Develops AF (%)'].append(np.round(af_perc_ind))

#%%
# Column names
columns = ['F', 'M'] + age_labels + labels[::-1] 

# Create dataframe
df_table = pd.DataFrame(table, index=columns)

# Transpose such that the grups are columns
df_table = df_table.T

if name == 'stratified':
    name_table = 'table.xlsx'
elif name == 'prevalence':
    name_table = 'suptable6.xlsx'

df_table.to_excel(manuscript_folder+name_table)

