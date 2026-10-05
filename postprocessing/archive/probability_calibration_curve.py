import pandas as pd
import numpy as np

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = "stratified"
test_file = folder + "dataset_test_" + name + ".xlsx"
df = pd.read_excel(test_file)

y_test = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob = np.load(output_file)["probs_test"].astype("float32")
THRESH = float(np.load(output_file)["thr"].astype("float32"))

df["y_prob"] = y_prob
df["y_true"] = y_test

AGE_EDGES      = [17, 60, 70, 80, 200]
AGE_LABELS     = ["18–60", "60–70", "70–80", "≥80"]

df["age_bin"] = pd.cut(
    df["age_at_last_ecg"],
    bins=AGE_EDGES,
    labels=AGE_LABELS,
    right=False,
    include_lowest=True
)

days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3])*days_per_year

labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"] = pd.cut(df['days_to_index'], bins=time_bins, labels=labels, right=False)
df["time_bin"] = pd.Categorical(df["time_bin"],
                                categories=labels[::-1],
                                ordered=True)

#%%
plt.figure()
colors = {"FEMALE": "gold", "MALE": "tab:blue"} 
# Subgroup
for group in df['Gender'].unique():
    
    color = colors[group]
    
    y_true_group = df[df['Gender'] == group]['y_true']
    y_pred_group = df[df['Gender'] == group]['y_prob']
    
    prob_true, prob_pred = calibration_curve(y_true_group, y_pred_group, n_bins=10)
    
    # Bepaal welke voorspelde kansen in welke bin vallen
    bin_indices = np.digitize(y_pred_group, bins=np.linspace(0, 1, 11)) - 1

    # Bereken het aantal patiënten per bin
    bin_counts = [np.sum(bin_indices == i) for i in range(10)]
    
    # Bereken standaardfout (SE) voor prob_true
    ses = [np.sqrt((prob_true[i] * (1 - prob_true[i])) / bin_counts[i])
           if bin_counts[i] > 0 else 0 for i in range(10)]
    
    # Plot alleen bins met data
    for i in range(10):
        if bin_counts[i] > 0:
            plt.errorbar(
                prob_pred[i],  # Gemiddelde voorspelde kans per bin
                prob_true[i],  # Waargenomen frequentie per bin
                yerr=ses[i],   # Standaardfout van de waargenomen frequentie
                fmt='o',
                color=colors[group],
                label=f'Gender: {group}' if i == 0 else ""
            )
    
    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal', color='k')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.legend()
    plt.title('Calibration curves by sex')
    plt.show()

#%%
plt.figure()
for group in df['age_bin'].unique():
    y_true_group = df[df['age_bin'] == group]['y_true']
    y_pred_group = df[df['age_bin'] == group]['y_prob']
    prob_true, prob_pred = calibration_curve(y_true_group, y_pred_group, n_bins=10)
    plt.plot(prob_pred, prob_true, label=f'Age: {group}')

plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.legend()
plt.title('Calibration curves by sex')
plt.show()

#%%
plt.figure()
for group in df['time_bin'].unique():
    y_true_group = df[df['time_bin'] == group]['y_true']
    y_pred_group = df[df['time_bin'] == group]['y_prob']
    prob_true, prob_pred = calibration_curve(y_true_group, y_pred_group, n_bins=10)
    plt.plot(prob_pred, prob_true, label=f'Timing: {group}')

plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.legend()
plt.title('Calibration curves by sex')
plt.show()


