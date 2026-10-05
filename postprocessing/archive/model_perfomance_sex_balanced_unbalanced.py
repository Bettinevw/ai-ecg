# performance model
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,               # Standaard lettergrootte
    'axes.titlesize': 14,          # Grootte van de assen-titels
    'axes.labelsize': 10,          # Grootte van de assen-labels
    'xtick.labelsize': 12,         # Grootte van de x-as ticklabels
    'ytick.labelsize': 12,         # Grootte van de y-as ticklabels
    'lines.linewidth': 3,          # Dikte van lijnen (bijv. medianen in violin plots)
    'grid.alpha': 0.7,             # Transparantie van de gridlijnen
})


def bootstrap_metrics(df, n_bootstraps=2000, seed=42):
    
    y_true = df["y_true"]
    y_prob = df["y_prob"] 
    y_pred = df["y_pred"]
    
    rng = np.random.RandomState(seed)
    
    # Initiate arrays
    bootstrapped_scores = []
    
    TP = []
    TN = []
    FP = []
    FN = []       
    
    ACC = []
    SENS = []
    SPEC = []
    PREC = [] 
    RECALL = []
    F1 = []

    for i in range(n_bootstraps):
        # sample with replacement
        indices = rng.randint(0, len(y_true), len(y_true))

        # AUROC is undefined if only one class is present
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue

        score = roc_auc_score(y_true.iloc[indices], y_prob.iloc[indices])
        
        tp = ((y_pred.iloc[indices] == 1) & (y_true.iloc[indices] == 1)).sum()
        tn = ((y_pred.iloc[indices] == 0) & (y_true.iloc[indices] == 0)).sum()
        fp = ((y_pred.iloc[indices] == 1) & (y_true.iloc[indices] == 0)).sum()
        fn = ((y_pred.iloc[indices] == 0) & (y_true.iloc[indices] == 1)).sum()
        
        acc  = (tp +tn) / (tp + tn + fn + fp)
        sens = tp / (tp + fn) 
        spec = tn / (tn + fp) 
        prec = tp / (tp + fp) 
        recall = sens
        f1 = 2*((prec*recall)/(prec+recall))
        
        bootstrapped_scores.append(score)
        
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        
        ACC.append(acc)
        SENS.append(sens)
        SPEC.append(spec)
        PREC.append(prec) 
        RECALL.append(recall)
        F1.append(f1)
        
    metric_dict = {'AUC' : bootstrapped_scores,
                   'TP' : TP,
                   'TN' : TN,
                   'FP' : FP,
                   'FN' : FN,
                   'ACC' : ACC,
                   'SENS': SENS,
                   'SPEC': SPEC,
                   'PREC': PREC,
                   'RECALL': RECALL,
                   'F1': F1}

    return metric_dict

#%% Dataset
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/MI_testset/"
manuscript_folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Manuscript/Revision/"
names = ['stratified', 'prevalence']
fig_labels = ['Unbalanced', 'Balanced'] 
colors = ['tab:purple', 'tab:pink']
plt.figure(figsize=(4, 4))
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal

table = {'AUROC': [], 
         'Sensitivity': [], 
         'Specificity': [], 
         'Precision': [],
         'Accuracy': [],
         'F1-score': []}

test_file   = folder + "dataset_test_" + name + ".xlsx"
df          = pd.read_excel(test_file)
X_test      = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test      = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob      = np.load(output_file)["probs_test"].astype("float32")
thr         = np.load(output_file)["thr"].astype("float32")

print(thr)

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

#%% Overall model performance
metric_dict     = bootstrap_metrics(df)

auc_low         = np.percentile(metric_dict['AUC'], 2.5)
auc_high        = np.percentile(metric_dict['AUC'], 97.5)
auc             = np.median(metric_dict['AUC'])

sens_low        = np.percentile(metric_dict['SENS'], 2.5)
sens_high       = np.percentile(metric_dict['SENS'], 97.5)
sens            = np.median(metric_dict['SENS'])

spec_low        = np.percentile(metric_dict['SPEC'], 2.5)
spec_high       = np.percentile(metric_dict['SPEC'], 97.5)
spec            = np.median(metric_dict['SPEC'])

prec_low        = np.percentile(metric_dict['PREC'], 2.5)
prec_high       = np.percentile(metric_dict['PREC'], 97.5)
prec            = np.median(metric_dict['PREC'])

f1_low          = np.percentile(metric_dict['F1'], 2.5)
f1_high         = np.percentile(metric_dict['F1'], 97.5)
f1              = np.median(metric_dict['F1'])

acc_low         = np.percentile(metric_dict['ACC'], 2.5)
acc_high        = np.percentile(metric_dict['ACC'], 97.5)
acc             = np.median(metric_dict['ACC'])

# Overall model performance
print(f'{fig_labels[i]}: Overall model performance')
print(f"AUROC = {auc:.2f} (95% CI {auc_low:.2f}–{auc_high:.2f})")
print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")
print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")
print(f"F1 = {f1:.2f} (95% CI {f1_low:.2f}–{f1_high:.2f})")


# Column names
table['AUROC'].append(f"{auc:.2f} [{auc_low:.2f}, {auc_high:.2f}]")
table['Sensitivity'].append(f"{sens:.2f} [{sens_low:.2f}, {sens_high:.2f}]")
table['Specificity'].append(f"{spec:.2f} [{spec_low:.2f}, {spec_high:.2f}]")
table['Precision'].append(f"{prec:.2f} [{prec_low:.2f}, {prec_high:.2f}]")
table['Accuracy'].append(f"{acc:.2f} [{acc_low:.2f}, {acc_high:.2f}]")
table['F1-score'].append(f"{f1:.2f} [{f1_low:.2f}, {f1_high:.2f}]")

columns = ['Unbalanced', 'Balanced']

# Create dataframe
df_table = pd.DataFrame(table, index=columns)

df_table.to_excel(manuscript_folder+'table2.xlsx')

#%% ROC-curve 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Plot    
plt.plot(fpr, tpr, label=f"{fig_labels[i]}: AUROC = {auc:.2f}", color=colors[i])    
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.xlim([0,1])
plt.ylim([0,1])
plt.tight_layout()
plt.show()