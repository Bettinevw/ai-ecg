# plot functions
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


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


def bootstrap_metrics(df, n_bootstraps=2000, seed=30): #42
    
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


def subgroup_plot(subgroup, order, labels, xlabel, significance, metric_list, metric_title):
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
                y_margin = 0.03  # Basisafstand tussen aanduidingen
                y_limit = 1.15
                y_step = 0.05 # minimal space between sign. lines
    
                # Sorteer significantie-aanduidingen op afstand (korte haakjes eerst)
                sign_sorted = sorted(
                    significance[metric_list[m]],
                    key=lambda x: abs(group_pos[x[0]] - group_pos[x[1]])
                )
                
                used_y_positions = []
    
                for k, (group1, group2, sig) in enumerate(sign_sorted):
                    x_left = group_pos[group1]
                    x_right = group_pos[group2]
    
                    # Bepaal de y-positie op basis van de hoogte van de twee violins
                    y_left = max(subgroup[group1][metric_list[m]])
                    y_right = max(subgroup[group2][metric_list[m]])
                    y_base = max(y_left, y_right) + y_margin
    
                    # Bepaal de y-positie voor het haakje (boven de hoogste violin)
                    y = y_base
                    iterations = 0
                    while iterations < 100:
                        # Controleer of y te dicht bij een bestaande positie ligt
                        too_close = any(abs(y - used_y) < y_step for used_y in used_y_positions)
                        if not too_close and y <= y_limit:
                            break
                        y += y_step
                        iterations += 1
                    
                    used_y_positions.append(y)
                    
                    # Teken het haakje met zijkantjes
                    ax[row, col].plot(
                        [x_left, x_right],
                        [y, y],
                        color="black",
                        linewidth=0.8,
                    )
                    ax[row, col].plot(
                        [x_left, x_left],
                        [y - y_margin, y],
                        color="black",
                        linewidth=0.8,
                    )
                    ax[row, col].plot(
                        [x_right, x_right],
                        [y - y_margin, y],
                        color="black",
                        linewidth=0.8,
                    )
    
                    # Voeg sterretjes toe
                    ax[row, col].text(
                        (x_left + x_right) / 2,
                        y- y_margin,
                        sig,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    # fig.tight_layout()
    plt.show()

