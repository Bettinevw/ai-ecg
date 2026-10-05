# Figure 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_functions import (bootstrap_metrics, calculate_significance, 
                            calculate_trend, subgroup_plot)

plt.rcParams.update({
    'font.size': 12,               # Standaard lettergrootte
    'axes.titlesize': 14,          # Grootte van de assen-titels
    'axes.labelsize': 12,          # Grootte van de assen-labels
    'xtick.labelsize': 10,         # Grootte van de x-as ticklabels
    'ytick.labelsize': 10,         # Grootte van de y-as ticklabels
    'lines.linewidth': 2,          # Dikte van lijnen (bijv. medianen in violin plots)
    'grid.alpha': 0.7,             # Transparantie van de gridlijnen
})

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = 'prevalence' #'stratified'

#%% Dataset
test_file   = folder + "dataset_test_" + name + ".xlsx"
df          = pd.read_excel(test_file)
X_test      = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test      = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob      = np.load(output_file)["probs_test"].astype("float32")
thr         = np.load(output_file)["thr"].astype("float32")

df["y_prob"]    = y_prob 

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
df["y_pred"] = (df["y_prob"] >= thr).astype(int)

# True labels
df["y_true"] = y_test[df.index]


#%% Subgroup analysis sex
metric_dict     = bootstrap_metrics(df)

subgroup_sex = {}
for gender in df["Gender"].unique():
    subgroup_df = df[df["Gender"] == gender]
    subgroup_sex[gender] = bootstrap_metrics(subgroup_df)
order_sex = ["FEMALE", "MALE"]  
label_sex = ["Female", "Male"]  
xlabel_sex = 'Sex'
significance_sex = calculate_significance(subgroup_sex, order_sex, metric_dict, alpha=0.05)

sens_low        = np.percentile(subgroup_sex['FEMALE']['SENS'], 2.5)
sens_high       = np.percentile(subgroup_sex['FEMALE']['SENS'], 97.5)
sens            = np.median(subgroup_sex['FEMALE']['SENS'])

spec_low        = np.percentile(subgroup_sex['FEMALE']['SPEC'], 2.5)
spec_high       = np.percentile(subgroup_sex['FEMALE']['SPEC'], 97.5)
spec            = np.median(subgroup_sex['FEMALE']['SPEC'])

prec_low        = np.percentile(subgroup_sex['FEMALE']['PREC'], 2.5)
prec_high       = np.percentile(subgroup_sex['FEMALE']['PREC'], 97.5)
prec            = np.median(subgroup_sex['FEMALE']['PREC'])

acc_low         = np.percentile(subgroup_sex['FEMALE']['ACC'], 2.5)
acc_high        = np.percentile(subgroup_sex['FEMALE']['ACC'], 97.5)
acc             = np.median(subgroup_sex['FEMALE']['ACC'])
print('Female')
print(f"acc = {acc:.2f} [95% CI {acc_low:.2f}–{acc_high:.2f}]")
print(f"sens = {sens:.2f} [95% CI {sens_low:.2f}–{sens_high:.2f}]" )
print(f"spec = {spec:.2f} [95% CI {spec_low:.2f}–{spec_high:.2f}]")
print(f"prec = {prec:.2f} [95% CI {prec_low:.2f}–{prec_high:.2f}]")

sens_low        = np.percentile(subgroup_sex['MALE']['SENS'], 2.5)
sens_high       = np.percentile(subgroup_sex['MALE']['SENS'], 97.5)
sens            = np.median(subgroup_sex['MALE']['SENS'])

spec_low        = np.percentile(subgroup_sex['MALE']['SPEC'], 2.5)
spec_high       = np.percentile(subgroup_sex['MALE']['SPEC'], 97.5)
spec            = np.median(subgroup_sex['MALE']['SPEC'])

prec_low        = np.percentile(subgroup_sex['MALE']['PREC'], 2.5)
prec_high       = np.percentile(subgroup_sex['MALE']['PREC'], 97.5)
prec            = np.median(subgroup_sex['MALE']['PREC'])

acc_low         = np.percentile(subgroup_sex['MALE']['ACC'], 2.5)
acc_high        = np.percentile(subgroup_sex['MALE']['ACC'], 97.5)
acc             = np.median(subgroup_sex['MALE']['ACC'])
print('Male')
print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")
print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")

#%%    
subgroup_age = {}
for age in df["age_bin"].unique():
    subgroup_df = df[df["age_bin"] == age]
    subgroup_age[age] = bootstrap_metrics(subgroup_df)
subgroup_age = {k: v for k, v in subgroup_age.items() if k == k}  
order_age = ['18–60', '60-70', '70–80', '≥80']
label_age = ['18–60', '60-70', '70–80', '≥80']
xlabel_age = 'Age bin [years]'

significance_age = calculate_significance(subgroup_age, order_age, metric_dict, alpha=0.05)

p_trend_age = calculate_trend(subgroup_age, order_age, metric_dict)

for i in order_age:
    sens_low        = np.percentile(subgroup_age[i]['SENS'], 2.5)
    sens_high       = np.percentile(subgroup_age[i]['SENS'], 97.5)
    sens            = np.median(subgroup_age[i]['SENS'])

    spec_low        = np.percentile(subgroup_age[i]['SPEC'], 2.5)
    spec_high       = np.percentile(subgroup_age[i]['SPEC'], 97.5)
    spec            = np.median(subgroup_age[i]['SPEC'])

    prec_low        = np.percentile(subgroup_age[i]['PREC'], 2.5)
    prec_high       = np.percentile(subgroup_age[i]['PREC'], 97.5)
    prec            = np.median(subgroup_age[i]['PREC'])

    acc_low         = np.percentile(subgroup_age[i]['ACC'], 2.5)
    acc_high        = np.percentile(subgroup_age[i]['ACC'], 97.5)
    acc             = np.median(subgroup_age[i]['ACC'])
    print(i)
    print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")
    print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
    print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
    print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")
    
#%% Plot subgroup analysis
metric_list = ['ACC', 'SENS', 'SPEC', 'PREC']
metric_title = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision']

subgroup_plot(subgroup_sex, order_sex, label_sex, xlabel_sex, significance_sex, metric_list, metric_title)
subgroup_plot(subgroup_age, order_age, label_age, xlabel_age, significance_age, metric_list, metric_title)