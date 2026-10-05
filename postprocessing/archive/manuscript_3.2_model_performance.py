# peformance model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, norm
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
import itertools
from scipy.stats import spearmanr

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
# name = 'stratified'
name = 'prevalence'

plt.close('all')



def calculate_trend(subgroup, order, metrics):
    """
    Bereken de Spearman rangcorrelatie voor trends in metrics over geordende groepen.

    Parameters:
    - subgroup: dictionary met bootstrapped data per groep (bijv. {'18-60': {'ACC': [array], ...}, ...})
    - order: volgorde van de groepen (bijv. ['18-60', '60-70', '70-80', '80+'])
    - metrics: lijst met metrics (bijv. ['ACC', 'PREC', 'SENS'])

    Returns:
    - trend: dictionary met Spearman resultaten per metric
    """
    trend = {metric: {} for metric in metrics}

    for metric in metrics:
        # Bereken de gemiddelde waarden per groep
        mean_values = [np.mean(subgroup[key][metric]) for key in order]

        # Groepsscores (bijv. 1, 2, 3, 4)
        group_scores = list(range(1, len(order) + 1))

        # Bereken Spearman rangcorrelatie
        rho, p_value = spearmanr(group_scores, mean_values)

        # Bepaal de richting van de trend
        if rho > 0:
            trend_direction = "stijgend"
        elif rho < 0:
            trend_direction = "dalend"
        else:
            trend_direction = "geen trend"

        trend[metric] = {
            'rho': rho,
            'p_value': p_value,
            'trend_direction': trend_direction,
            'mean_values': mean_values
        }

    return trend


def calculate_significance(subgroup, order, metrics, alpha=0.05):
    """
    Bereken significantie tussen groepen voor proporties (precision, sensitiviteit, specificiteit, accuracy),
    gebruikmakend van de meanen van bootstrapped distributies.

    Parameters:
    - subgroup: dictionary met bootstrapped data per groep (bijv. {'Group1': {'TP': [array], 'FP': [array], ...}, ...})
    - order: volgorde van de groepen
    - metrics: lijst met metrics (bijv. ['PREC', 'SENS', 'SPEC', 'ACC'])
    - alpha: significantieniveau (standaard 0.05)

    Returns:
    - significance: dictionary met significantie-informatie
    """
    significance = {metric: [] for metric in metrics}

    for metric in metrics:
        # Bereken meanen voor successes en nobs
        mean_successes = []
        mean_nobs = []

        for key in order:
            if metric == 'PREC':
                # Precision = mean(TP) / (mean(TP) + mean(FP))
                mean_tp = np.mean(subgroup[key]['TP'])
                mean_fp = np.mean(subgroup[key]['FP'])
                mean_successes.append(mean_tp)
                mean_nobs.append(mean_tp + mean_fp)

            elif metric == 'SENS':
                # Sensitiviteit = mean(TP) / (mean(TP) + mean(FN))
                mean_tp = np.mean(subgroup[key]['TP'])
                mean_fn = np.mean(subgroup[key]['FN'])
                mean_successes.append(mean_tp)
                mean_nobs.append(mean_tp + mean_fn)

            elif metric == 'SPEC':
                # Specificiteit = mean(TN) / (mean(TN) + mean(FP))
                mean_tn = np.mean(subgroup[key]['TN'])
                mean_fp = np.mean(subgroup[key]['FP'])
                mean_successes.append(mean_tn)
                mean_nobs.append(mean_tn + mean_fp)

            elif metric == 'ACC':
                # Accuracy = (mean(TP) + mean(TN)) / (mean(TP) + mean(TN) + mean(FP) + mean(FN))
                mean_tp = np.mean(subgroup[key]['TP'])
                mean_tn = np.mean(subgroup[key]['TN'])
                mean_fp = np.mean(subgroup[key]['FP'])
                mean_fn = np.mean(subgroup[key]['FN'])
                mean_successes.append(mean_tp + mean_tn)
                mean_nobs.append(mean_tp + mean_tn + mean_fp + mean_fn)

        
        if len(mean_successes) > 0:
            # Bereken p-waarden voor alle paren groepen
            p_values = []
            group_pairs = list(combinations(enumerate(order), 2))

            for (i, _), (j, _) in group_pairs:
                # Rond af naar hele getallen (proportions_ztest verwacht integers)
                count = np.array([int(np.round(mean_successes[i])), int(np.round(mean_successes[j]))])
                nobs_array = np.array([int(np.round(mean_nobs[i])), int(np.round(mean_nobs[j]))])

                # Voer proportions_ztest uit en sla de p-waarde op
                _, p_val = proportions_ztest(count, nobs_array)
                p_values.append(p_val)

            # Bepaal significantieniveau voor elk groepspaar
            for idx, ((i, _), (j, _)) in enumerate(group_pairs):
                if p_values[idx] < 0.001:
                    sig = '***'
                elif p_values[idx] < 0.01:
                    sig = '**'
                elif p_values[idx] < alpha:
                    sig = '*'
                else:
                    sig = None

                if sig is not None:
                    significance[metric].append((order[i], order[j], sig))

    return significance

#%%
def subgroup_plot(subgroup, order, labels, xlabel, significance):
    fig, ax = plt.subplots(2, 2, figsize=(1.5 * len(subgroup.keys()), 5), sharey=True, sharex=True)
    ax[0, 0].set_ylim([0.2, 1.22])
    
    group_pos = {group: idx + 1 for idx, group in enumerate(order)}
    
    for m in range(len(metric_list)):
        row = m // 2
        col = m % 2

        data = [subgroup[key][metric_list[m]] for key in order]

        parts = ax[row, col].violinplot(data, showmeans=False, showmedians=True, showextrema=True)

        # Stel de kleur in voor de violins
        for pc in parts['bodies']:
            pc.set_facecolor('tab:purple')
            pc.set_edgecolor('tab:purple')
            pc.set_alpha(0.7)

        # Stel de kleur in voor de centrale lijn (median/mean)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('tab:purple')
                vp.set_linewidth(1)

        ax[row, col].set_title(metric_title[m])
        ax[row, col].set_xticks([y + 1 for y in range(len(data))], labels=labels)
        ax[row, col].yaxis.grid(True)
        
        
        if not xlabel == 'Time to index ECG bin':

            if metric_list[m] in significance:
                y_margin = 0.02  # Basisafstand tussen aanduidingen
    
                # Sorteer significantie-aanduidingen op afstand (korte haakjes eerst)
                sign_sorted = sorted(
                    significance[metric_list[m]],
                    key=lambda x: abs(group_pos[x[0]] - group_pos[x[1]])
                )
    
                for k, (group1, group2, sig) in enumerate(sign_sorted):
                    x_left = group_pos[group1]
                    x_right = group_pos[group2]
    
                    # Bepaal de y-positie op basis van de hoogte van de twee violins
                    y_left = max(subgroup[group1][metric_list[m]])
                    y_right = max(subgroup[group2][metric_list[m]])
                    y_max = max(y_left, y_right)  # Hoogste waarde van de twee violins
    
                    # Bepaal de y-positie voor het haakje (boven de hoogste violin)
                    y = y_max + (2 + 2.5 * k) * y_margin
    
                    # Teken het haakje met zijkantjes
                    ax[row, col].plot(
                        [x_left, x_right],
                        [y + y_margin, y + y_margin],
                        color="black",
                        linewidth=0.8,
                    )
                    ax[row, col].plot(
                        [x_left, x_left],
                        [y, y + y_margin],
                        color="black",
                        linewidth=0.8,
                    )
                    ax[row, col].plot(
                        [x_right, x_right],
                        [y, y + y_margin],
                        color="black",
                        linewidth=0.8,
                    )
    
                    # Voeg sterretjes toe
                    ax[row, col].text(
                        (x_left + x_right) / 2,
                        y,
                        sig,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    fig.tight_layout()
    plt.show()
 #%%   


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

#%% Sub group plot sex
def subgroup_plot_interaction(subgroup, order, labels, xlabel,
                             significance, colors, metric_list, metric_title):
    
    fig, ax = plt.subplots(2, 2, figsize=(1.5 * len(subgroup.keys()), 5), sharey=True, sharex=True)
    ax[0, 0].set_ylim([0.0, 1.1])
    
    # Positions voor de violins
    positions = [i + 1 for i in range(len(order))]
    male_positions = [pos + 0.2 for pos in positions]
    female_positions = [pos - 0.2 for pos in positions]

    for m in range(len(metric_list)):
        row = m // 2
        col = m % 2

        data_female = [subgroup[key]['FEMALE'][metric_list[m]] for key in order]
        data_male = [subgroup[key]['MALE'][metric_list[m]] for key in order]

        parts_f = ax[row, col].violinplot(data_female, positions=female_positions, showmeans=False, showmedians=True, showextrema=True)
        parts_m = ax[row, col].violinplot(data_male, positions=male_positions,showmeans=False, showmedians=True, showextrema=True)
        
        bracket_count = {(row, col, par): 0 for row in range(2) 
                         for col in range(2) for par in order}
        
        # Stel de kleur in voor de violins
        for pc in parts_f['bodies']:
            pc.set_facecolor(colors[0])
            pc.set_edgecolor(colors[0])
            pc.set_alpha(0.7)

        # Stel de kleur in voor de centrale lijn (median/mean)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts_f:
                vp = parts_f[partname]
                vp.set_edgecolor(colors[0])
                vp.set_linewidth(1)
                
        for pc in parts_m['bodies']:
            pc.set_facecolor(colors[1])
            pc.set_edgecolor(colors[1])
            pc.set_alpha(0.7)

        # Stel de kleur in voor de centrale lijn (median/mean)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts_m:
                vp = parts_m[partname]
                vp.set_edgecolor(colors[1])
                vp.set_linewidth(1)

        ax[row, col].set_title(metric_title[m])
        ax[row, col].set_xticks(positions, labels=labels)
        ax[row, col].yaxis.grid(True)
        
        y_margin = 0.02
        # Check of er significantie-data is voor deze metric
        for idx, par in enumerate(order):
            if  len(significance[par][metric_list[m]]) > 0:
                # Bepaal y-positie gebaseerd op de hoogste violin
                y_max = max(
                    max(data_female[idx]) if data_female[idx] else 0,
                    max(data_male[idx]) if data_male[idx] else 0
                )

                # Stack haakjes verticaal
                k = bracket_count[(row, col, par)]
                y = y_max + (2 + 2*k) * y_margin
                bracket_count[(row, col, par)] += 1

                # Teken het haakje tussen man en vrouw
                ax[row, col].plot(
                    [male_positions[idx], male_positions[idx], female_positions[idx], female_positions[idx]],
                    [y, y + y_margin, y + y_margin, y],
                    color="black",
                    linewidth=0.8,
                )

                # Voeg sterretjes toe voor p-waarde
                star = significance[par][metric_list[m]][0][2]
                ax[row, col].text(
                    (male_positions[idx] + female_positions[idx]) / 2,  # Midden tussen man/vrouw
                    y + y_margin,
                    star,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    # Legenda toe
    ax[0,0].legend(
        [plt.Line2D([0], [0], color=colors[0], lw=2),
         plt.Line2D([0], [0], color=colors[1], lw=2)],
        ['Female', 'Male'],
        loc='lower right'
    )

    fig.tight_layout()
    plt.show()


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
print('Overall model performance')
print(f"AUROC = {auc:.2f} (95% CI {auc_low:.2f}–{auc_high:.2f})")
print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")
print(f"F1 = {f1:.2f} (95% CI {f1_low:.2f}–{f1_high:.2f})")
print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")

#%% Subgroup analysis sex
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

    
#%%
subgroup_time = {}
for time in df["time_bin"].unique():
    subgroup_df = df[df["time_bin"] == time]
    subgroup_time[time] = bootstrap_metrics(subgroup_df)
order_time = ['2-3y', '1-2y', '6–12m', '3–6m', '1–3m', '1w–1m','≤1w'] 
label_time = ['2-3y', '1-2y', '6–12m', '3–6m', '1–3m', '1w–1m','≤1w'] 
xlabel_time = 'Time to index ECG bin'

significance_time = calculate_significance(subgroup_time, order_time, metric_dict, alpha=0.05)
p_trend_time = calculate_trend(subgroup_time, order_time, metric_dict)



for i in order_time:
    sens_low        = np.percentile(subgroup_time[i]['SENS'], 2.5)
    sens_high       = np.percentile(subgroup_time[i]['SENS'], 97.5)
    sens            = np.median(subgroup_time[i]['SENS'])

    spec_low        = np.percentile(subgroup_time[i]['SPEC'], 2.5)
    spec_high       = np.percentile(subgroup_time[i]['SPEC'], 97.5)
    spec            = np.median(subgroup_time[i]['SPEC'])

    prec_low        = np.percentile(subgroup_time[i]['PREC'], 2.5)
    prec_high       = np.percentile(subgroup_time[i]['PREC'], 97.5)
    prec            = np.median(subgroup_time[i]['PREC'])

    acc_low         = np.percentile(subgroup_time[i]['ACC'], 2.5)
    acc_high        = np.percentile(subgroup_time[i]['ACC'], 97.5)
    acc             = np.median(subgroup_time[i]['ACC'])
    print(i)
    print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")
    print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
    print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
    print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")


#%% Plot subgroup analysis
metric_list = ['ACC', 'SENS', 'SPEC', 'PREC']
metric_title = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision']

subgroup_plot(subgroup_sex, order_sex, label_sex, xlabel_sex, significance_sex)
subgroup_plot(subgroup_age, order_age, label_age, xlabel_age, significance_age)
subgroup_plot(subgroup_time, order_time, label_time, xlabel_time, significance_time)

#%% Sex age interaction
subgroup_sex_age = {}
for age in df["age_bin"].unique():
    subgroup_sex_age[age] = {}
    for sex in df["Gender"].unique():  # 
        subgroup_df = df[(df["Gender"] == sex) & (df["age_bin"] == age)]
        subgroup_sex_age[age][sex] = bootstrap_metrics(subgroup_df)
subgroup_sex_age = {k: v for k, v in subgroup_sex_age.items() if k == k} 

# Calulate significance
significance_sex_age = {}
for age in subgroup_sex_age.keys():
    significance_sex_age[age] = calculate_significance(
        subgroup_sex_age[age], order_sex, metric_dict, alpha=0.05
    )
    
subgroup_plot_interaction(subgroup_sex_age, order_age, label_age, xlabel_age,
                          significance_sex_age, ['tab:orange', 'tab:blue'], metric_list, metric_title)
#%% Sex age interaction print
for i in order_age:
    for g in order_sex:
        sens_low        = np.percentile(subgroup_sex_age[i][g]['SENS'], 2.5)
        sens_high       = np.percentile(subgroup_sex_age[i][g]['SENS'], 97.5)
        sens            = np.median(subgroup_sex_age[i][g]['SENS'])
    
        spec_low        = np.percentile(subgroup_sex_age[i][g]['SPEC'], 2.5)
        spec_high       = np.percentile(subgroup_sex_age[i][g]['SPEC'], 97.5)
        spec            = np.median(subgroup_sex_age[i][g]['SPEC'])
    
        prec_low        = np.percentile(subgroup_sex_age[i][g]['PREC'], 2.5)
        prec_high       = np.percentile(subgroup_sex_age[i][g]['PREC'], 97.5)
        prec            = np.median(subgroup_sex_age[i][g]['PREC'])
    
        acc_low         = np.percentile(subgroup_sex_age[i][g]['ACC'], 2.5)
        acc_high        = np.percentile(subgroup_sex_age[i][g]['ACC'], 97.5)
        acc             = np.median(subgroup_sex_age[i][g]['ACC'])
        print(i)
        print(g)
        print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")
        print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
        print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
        print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")

#%% Sex time interaction
subgroup_sex_time = {}
for time in df["time_bin"].unique():
    subgroup_sex_time[time] = {}
    for sex in df["Gender"].unique():  # 
        subgroup_df = df[(df["Gender"] == sex) & (df["time_bin"] == time)]
        subgroup_sex_time[time][sex] = bootstrap_metrics(subgroup_df)
subgroup_sex_time = {k: v for k, v in subgroup_sex_time.items() if k == k} 

# Calulate significance
significance_sex_time = {}
for time in subgroup_sex_time.keys():
    significance_sex_time[time] = calculate_significance(
        subgroup_sex_time[time], order_sex, metric_dict, alpha=0.05
    )
    
subgroup_plot_interaction(subgroup_sex_time, order_time, label_time, xlabel_time,
                          significance_sex_time, ['tab:orange', 'tab:blue'], metric_list, metric_title)

#%% Sex time interaction print
for i in order_time:
    for g in order_sex:
        sens_low        = np.percentile(subgroup_sex_time[i][g]['SENS'], 2.5)
        sens_high       = np.percentile(subgroup_sex_time[i][g]['SENS'], 97.5)
        sens            = np.median(subgroup_sex_time[i][g]['SENS'])
    
        spec_low        = np.percentile(subgroup_sex_time[i][g]['SPEC'], 2.5)
        spec_high       = np.percentile(subgroup_sex_time[i][g]['SPEC'], 97.5)
        spec            = np.median(subgroup_sex_time[i][g]['SPEC'])
    
        prec_low        = np.percentile(subgroup_sex_time[i][g]['PREC'], 2.5)
        prec_high       = np.percentile(subgroup_sex_time[i][g]['PREC'], 97.5)
        prec            = np.median(subgroup_sex_time[i][g]['PREC'])
    
        acc_low         = np.percentile(subgroup_sex_time[i][g]['ACC'], 2.5)
        acc_high        = np.percentile(subgroup_sex_time[i][g]['ACC'], 97.5)
        acc             = np.median(subgroup_sex_time[i][g]['ACC'])
        print(i)
        print(g)
        print(f"acc = {acc:.2f} (95% CI {acc_low:.2f}–{acc_high:.2f})")
        print(f"sens = {sens:.2f} (95% CI {sens_low:.2f}–{sens_high:.2f})" )
        print(f"spec = {spec:.2f} (95% CI {spec_low:.2f}–{spec_high:.2f})")
        print(f"prec = {prec:.2f} (95% CI {prec_low:.2f}–{prec_high:.2f})")