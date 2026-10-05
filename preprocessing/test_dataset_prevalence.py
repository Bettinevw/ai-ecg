import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
random.seed(30)

# Load the dataset
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
manuscript_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Manuscript/"
name = 'stratified'

dataset_test_file = folder + "dataset_test_" + name + ".xlsx"
df = pd.read_excel(dataset_test_file)

# Define time and age bins
days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3]) * days_per_year
time_labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"] = pd.cut(df['days_to_index'], bins=time_bins, labels=time_labels, right=False)

age_bins = [17, 60, 70, 80, 200]
age_labels = ["18–60", "60-70", "70–80", "≥80"]
df["age_bin"] = pd.cut(
    df["PatientAge"], bins=age_bins, labels=age_labels,
    right=False, include_lowest=True
)

# Target prevalence
ideal_prev = 0.30
remove_pin_list = []

# Combined loop for time_bin, age_bin, and Gender
for time in time_labels:
    for age in age_labels:
        for gender in ['FEMALE', 'MALE']:
            # Filter the subgroup
            subgroup_df = df[
                (df["Gender"] == gender) &
                (df["time_bin"] == time) &
                (df["age_bin"] == age)
            ]

            # Skip if subgroup is empty
            if len(subgroup_df) == 0:
                continue

            # Calculate current AF prevalence
            af_prev = len(subgroup_df[subgroup_df['Group'] == 'AF']) / len(subgroup_df)

            if af_prev > ideal_prev:
                # Remove AF cases
                nr_af_reduction = int(np.round(
                    (len(subgroup_df[subgroup_df['Group'] == 'AF']) - ideal_prev * len(subgroup_df)) /
                    (1 - ideal_prev)
                ))
                subgroup_af = subgroup_df[subgroup_df['Group'] == 'AF']
                af_pins = subgroup_af['PIN'].values
                pin_reduction = random.sample(
                    list(af_pins),
                    min(nr_af_reduction, len(af_pins))
                )
            else:
                # Remove Control cases
                nr_ctrl_reduction = int(np.round(
                    len(subgroup_df) - (len(subgroup_df[subgroup_df['Group'] == 'AF']) / ideal_prev)
                ))
                subgroup_ctrl = subgroup_df[subgroup_df['Group'] == 'Control']
                ctrl_pins = subgroup_ctrl['PIN'].values
                pin_reduction = random.sample(
                    list(ctrl_pins),
                    min(nr_ctrl_reduction, len(ctrl_pins))
                )

            remove_pin_list.extend(pin_reduction)

# Remove duplicates from remove_pin_list
remove_pin_list = list(set(remove_pin_list))

# Apply filtering once
df_prevalence = df[~df['PIN'].isin(remove_pin_list)]

# Save the result
dataset_test_file_prevalence = folder + "dataset_test_prevalence.xlsx"
df_prevalence.to_excel(dataset_test_file_prevalence, index=False)

#%%
# Old results
X_test      = np.load(folder+"test_dataset_stratified.npy")
y_test      = np.load(folder+"test_outcome_stratified.npz")
prob        = np.load(folder + "prob_stratified.npz")

# New results
pins = np.char.decode(X_test['PIN'], encoding='utf-8').astype(int)
ecgs = X_test['ecg']

# Filter de indices waar PIN niet in remove_pin_list zit
mask = ~np.isin(pins, remove_pin_list)

# Pas de mask toe op de data
filtered_ecgs = ecgs[mask]
filtered_pins = pins[mask]

np.savez(folder+"test_dataset_prevalence.npy", ecg=filtered_ecgs, PIN=filtered_pins)


# New results
pins_out = y_test['PIN'].astype(int)
outcome = y_test['outcome']
out_sex = y_test['out_sex']
out_age = y_test['out_age']

# Filter de indices waar PIN niet in remove_pin_list zit
mask_out = ~np.isin(pins_out, remove_pin_list)

# Apply mask
filtered_pins_out = pins_out[mask_out]
filtered_outcome = outcome[mask_out]
filtered_out_sex = out_sex[mask_out]
filtered_out_age = out_age[mask_out]


np.savez(folder+"test_outcome_prevalence.npz", outcome=filtered_outcome, PIN=filtered_pins_out, 
         out_sex=filtered_out_sex, out_age=filtered_out_age)

# Apply mask
probs_val = prob['probs_val']
probs_test = prob['probs_test']
t_best = prob['thr']
y_hat = prob['y_hat']

filtered_probs_test = probs_test[mask_out]
filtered_y_hat = y_hat[mask_out]

np.savez(folder + "prob_prevalence.npz", probs_val=probs_val, 
         probs_test=filtered_probs_test, 
         thr=t_best, y_hat=filtered_y_hat)