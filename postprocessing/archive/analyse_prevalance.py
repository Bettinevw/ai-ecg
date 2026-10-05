import pandas as pd
import numpy as np

# Load the dataset
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
manuscript_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Manuscript/Revision/"

dataset_test_file = folder + "dataset_val.xlsx"
df = pd.read_excel(dataset_test_file)

# Define time and age bins
days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3]) * days_per_year
labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"] = pd.cut(df['days_to_index'], bins=time_bins, labels=labels, right=False)
df["time_bin"] = pd.Categorical(df["time_bin"], categories=labels[::-1], ordered=True)

age_bins = [17, 60, 70, 80, 200]
age_labels = ["18–60", "60-70", "70–80", "≥80"]
df["age_bin"] = pd.cut(
    df["PatientAge"], bins=age_bins, labels=age_labels,
    right=False, include_lowest=True
)

# Initialize the table
table = {
    'Nr of individuals': [],
    'Develops AF (%)': [],
    'Sex (% female)': []
}

# Track the multi-index rows
row_labels = []

# Add overall gender groups (F, M)
for gender in ['FEMALE', 'MALE']:
    subgroup_df = df[df["Gender"] == gender]
    af_perc_ind = len(subgroup_df[subgroup_df['Group'] == 'AF']) / len(subgroup_df) * 100
    female_perc = len(subgroup_df[subgroup_df['Gender'] == 'FEMALE']) / len(subgroup_df) * 100

    table['Nr of individuals'].append(len(subgroup_df))
    table['Develops AF (%)'].append(np.round(af_perc_ind, 1))
    table['Sex (% female)'].append(np.round(female_perc, 1))
    row_labels.append((gender, 'Overall'))

# Add age groups
for age in age_labels:
    subgroup_df = df[df["age_bin"] == age]
    af_perc_ind = len(subgroup_df[subgroup_df['Group'] == 'AF']) / len(subgroup_df) * 100
    female_perc = len(subgroup_df[subgroup_df['Gender'] == 'FEMALE']) / len(subgroup_df) * 100

    # Add the overall age group
    table['Nr of individuals'].append(len(subgroup_df))
    table['Develops AF (%)'].append(np.round(af_perc_ind, 1))
    table['Sex (% female)'].append(np.round(female_perc, 1))
    row_labels.append((age, 'Overall'))

    # Add subgroups for F and M within this age group
    for gender in ['FEMALE', 'MALE']:
        gender_subgroup = subgroup_df[subgroup_df["Gender"] == gender]
        if len(gender_subgroup) > 0:
            af_perc_gender = len(gender_subgroup[gender_subgroup['Group'] == 'AF']) / len(gender_subgroup) * 100
            female_perc_gender = len(gender_subgroup[gender_subgroup['Gender'] == 'FEMALE']) / len(gender_subgroup) * 100

            table['Nr of individuals'].append(len(gender_subgroup))
            table['Develops AF (%)'].append(np.round(af_perc_gender, 1))
            table['Sex (% female)'].append(np.round(female_perc_gender, 1))
            row_labels.append((age, 'F' if gender == 'FEMALE' else 'M'))

# Add time intervals
for time in labels[::-1]:
    subgroup_df = df[df["time_bin"] == time]
    af_perc_ind = len(subgroup_df[subgroup_df['Group'] == 'AF']) / len(subgroup_df) * 100
    female_perc = len(subgroup_df[subgroup_df['Gender'] == 'FEMALE']) / len(subgroup_df) * 100

    # Add the overall time interval
    table['Nr of individuals'].append(len(subgroup_df))
    table['Develops AF (%)'].append(np.round(af_perc_ind, 1))
    table['Sex (% female)'].append(np.round(female_perc, 1))
    row_labels.append((time, 'Overall'))

    # Add subgroups for F and M within this time interval
    for gender in ['FEMALE', 'MALE']:
        gender_subgroup = subgroup_df[subgroup_df["Gender"] == gender]
        if len(gender_subgroup) > 0:
            af_perc_gender = len(gender_subgroup[gender_subgroup['Group'] == 'AF']) / len(gender_subgroup) * 100
            female_perc_gender = len(gender_subgroup[gender_subgroup['Gender'] == 'FEMALE']) / len(gender_subgroup) * 100

            table['Nr of individuals'].append(len(gender_subgroup))
            table['Develops AF (%)'].append(np.round(af_perc_gender, 1))
            table['Sex (% female)'].append(np.round(female_perc_gender, 1))
            row_labels.append((time, 'F' if gender == 'FEMALE' else 'M'))

# Create the DataFrame with multi-index
df_table = pd.DataFrame(table, index=pd.MultiIndex.from_tuples(row_labels, names=['Group', 'Subgroup']))

# Save to Excel
# df_table.to_excel(manuscript_folder + 'table2.xlsx')