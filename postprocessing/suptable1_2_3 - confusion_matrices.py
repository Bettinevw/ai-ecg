# Supplementary Material - Confusion matrices

import numpy as np
import pandas as pd

def confusion_matrix(df):
    # Confusion matrix
    tp = ((df["y_pred"] == 1) & (df["y_true"] == 1)).sum()
    tn = ((df["y_pred"] == 0) & (df["y_true"] == 0)).sum()
    fp = ((df["y_pred"] == 1) & (df["y_true"] == 0)).sum()
    fn = ((df["y_pred"] == 0) & (df["y_true"] == 1)).sum()
    
    return tp, tn, fp, fn


#%% Load data
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = 'prevalence' #stratified

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

#%% overall
tp, tn, fp, fn = confusion_matrix(df)
print("Overall")
print(len(df))
print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
print("TP:", np.round(tp/len(df),2), "FP:", np.round(fp/len(df),2), 
      "FN:", np.round(fn/len(df),2), "TN:", np.round(tn/len(df),2))

#%% Sex
genders = ["FEMALE", "MALE"]
for gender in genders:
    subgroup_df = df[df["Gender"] == gender]
    tp, tn, fp, fn = confusion_matrix(subgroup_df)
    
    print(gender)
    print(len(subgroup_df))
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("TP:", np.round(tp/len(subgroup_df),2), "FP:", np.round(fp/len(subgroup_df),2), 
          "FN:", np.round(fn/len(subgroup_df),2), "TN:", np.round(tn/len(subgroup_df),2))

#%% Age
for age in age_labels:
    subgroup_df = df[df["age_bin"] == age]
    tp, tn, fp, fn = confusion_matrix(subgroup_df)
    
    print(age)
    print(len(subgroup_df))
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("TP:", np.round(tp/len(subgroup_df),2), "FP:", np.round(fp/len(subgroup_df),2), 
          "FN:", np.round(fn/len(subgroup_df),2), "TN:", np.round(tn/len(subgroup_df),2))
 
#%% Time
for time in labels:
    subgroup_df = df[df["time_bin"] == time]
    tp, tn, fp, fn = confusion_matrix(subgroup_df)
    
    print(time)
    print(len(subgroup_df))
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("TP:", np.round(tp/len(subgroup_df),2), "FP:", np.round(fp/len(subgroup_df),2), 
          "FN:", np.round(fn/len(subgroup_df),2), "TN:", np.round(tn/len(subgroup_df),2))
    