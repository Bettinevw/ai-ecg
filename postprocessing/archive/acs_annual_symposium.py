import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/archive/"

def acc_ci(g):
    #Je kunt dan per bin een 95% betrouwbaarheidsinterval (CI) schatten met een 
    #binomiale test, en kijken of de CIs overlappen.
    correct = (g["y_pred"] == g["y_true"]).sum()
    total   = len(g)
    acc     = correct / total
    ci_low, ci_high = proportion_confint(correct, total, alpha=0.05, method="wilson")
    
    TP = ((g["y_pred"] == 1) & (g["y_true"] == 1)).sum()
    TN = ((g["y_pred"] == 0) & (g["y_true"] == 0)).sum()
    FP = ((g["y_pred"] == 1) & (g["y_true"] == 0)).sum()
    FN = ((g["y_pred"] == 0) & (g["y_true"] == 1)).sum()
    
    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    
    return pd.Series({
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "correct": correct,
        "total": total,
        "ci_low": ci_low,
        "ci_high": ci_high
    })

def compare_groups(rowA, rowB):
    count = [rowA["correct"], rowB["correct"]]       # aantallen correcte voorspellingen
    nobs =  [rowA["total"],   rowB["total"]]         # totale aantallen
    stat, pval = proportions_ztest(count, nobs)
    return stat, pval

def p_to_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

color_map = {
    "FEMALE": "gold",
    "MALE":   "tab:blue",
    "ALL":    "#9b59b6"   # paars
}

# Load data
name = "all_ecgs"

if name == 'all_ecgs' or name == 'all_ecgs_strata':
    time_col = "DaysBetweenECGs"
    test_file      = folder+ "dataset_test_"+name+".xlsx"
    df      = pd.read_excel(test_file)
    X_test  = np.load(folder+"test_dataset_" + name + ".npz")["ecg"].astype("float32")
    y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
    output_file = folder + "prob_"+name+".npz"
    y_prob = np.load(output_file)["probs_test"].astype("float32")
elif name == "balancing_by_matching":
    time_col = "DaysBetweenECGs"
    test_file      = folder+ "dataset_test_"+name+".xlsx"
    df      = pd.read_excel(test_file)
    X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
    y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
    output_file = folder + "prob_"+name+".npz"
    y_prob = np.load(output_file)["probs_test"].astype("float32")
    
else:
    time_col = "days_to_index"
    test_file      = folder+ "dataset_test_"+name+".xlsx"
    df      = pd.read_excel(test_file)
    X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
    y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
    output_file = folder + "prob_"+name+".npz"
    y_prob = np.load(output_file)["probs_test"].astype("float32")
    
df = df.copy()
df["y_prob"] = y_prob 

days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3])*days_per_year

labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"] = pd.cut(df[time_col], bins=time_bins, labels=labels, right=False)
df["time_bin"] = pd.Categorical(df["time_bin"],
                                categories=labels[::-1],
                                ordered=True)

age_bins   = [17, 60, 70, 80, 200]
age_labels = ["18–60","60-70", "70–80", "≥80"]
df["age_bin"] = pd.cut(
    df["PatientAge"], bins=age_bins, labels=age_labels,
    right=False, include_lowest=True
)

threshold = 0.60
df["y_pred"] = (df["y_prob"] >= threshold).astype(int)

# True labels
df["y_true"] = y_test[df.index]

# Accuracy per sex and age_bin
acc_by_sex_age_ci = df.groupby(["Gender", "age_bin"]).apply(acc_ci)
# Accuracy per sex and time_bin
acc_by_sex_time_ci = df.groupby(["Gender", "time_bin"]).apply(acc_ci)

# EXTRA: Accuracy per age_bin (M+F samen) en per time_bin (M+F samen)
acc_by_age_ci  = df.groupby("age_bin").apply(acc_ci)
acc_by_time_ci = df.groupby("time_bin").apply(acc_ci)

# ------------------------------------------------------------------
# AGE-PLOT: M vs F + ALL (paars) + significatie tussen paarse balken
# ------------------------------------------------------------------

# M vs F per age-bin (zoals je al had)
results = []

for age in acc_by_sex_age_ci.index.get_level_values("age_bin").unique():
    FEMALE = acc_by_sex_age_ci.loc[("FEMALE", age)]
    MALE   = acc_by_sex_age_ci.loc[("MALE", age)]
    stat, pval = compare_groups(MALE, FEMALE)
    results.append({
        "age_bin": age,
        "acc_male": MALE["accuracy"],
        "acc_female": FEMALE["accuracy"],
        "difference": MALE["accuracy"] - FEMALE["accuracy"],
        "p_value": pval
    })

signif_df = pd.DataFrame(results).sort_values("age_bin")
signif_df["sig"] = signif_df["p_value"].apply(p_to_star)
print(signif_df)

# Significantie tussen ALL (paars) balken: adjacent age-bins
results_age_overall = []
age_levels = acc_by_age_ci.index.to_list()

for i in range(len(age_levels) - 1):
    ageA = age_levels[i]
    ageB = age_levels[i+1]
    rowA = acc_by_age_ci.loc[ageA]
    rowB = acc_by_age_ci.loc[ageB]
    stat, pval = compare_groups(rowA, rowB)
    results_age_overall.append({
        "age_A": ageA,
        "age_B": ageB,
        "acc_A": rowA["accuracy"],
        "acc_B": rowB["accuracy"],
        "p_value": pval
    })

signif_age_overall = pd.DataFrame(results_age_overall)
signif_age_overall["sig"] = signif_age_overall["p_value"].apply(p_to_star)
print(signif_age_overall)

# Plot
df_plot = acc_by_sex_age_ci.reset_index()

fig, ax = plt.subplots(figsize=(5, 4))

age_bins_plot = df_plot["age_bin"].unique()
genders = ["FEMALE", "MALE"]

x = np.arange(len(age_bins_plot))
width = 0.25  # smaller: 3 bars per bin

age_pos = {age: i for i, age in enumerate(age_bins_plot)}

# Balken voor FEMALE en MALE
for i, gender in enumerate(genders):
    subset = df_plot[df_plot["Gender"] == gender].set_index("age_bin").loc[age_bins_plot]

    means = subset["accuracy"].values
    ci_low = subset["ci_low"].values
    ci_high = subset["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])

    # posities: FEMALE links (i=0), MALE midden (i=1)
    x_pos = x + (i - 0.5) * width

    ax.bar(
        x_pos,
        means,
        width,
        label=gender.capitalize(),
        color=color_map[gender],
        yerr=yerr,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

# Paarse balken voor ALL
overall = acc_by_age_ci.loc[age_bins_plot]
means_all = overall["accuracy"].values
ci_low_all = overall["ci_low"].values
ci_high_all = overall["ci_high"].values
yerr_all = np.vstack([means_all - ci_low_all, ci_high_all - means_all])

# posities: ALL rechts
x_pos_all = x + (1.5) * width

ax.bar(
    x_pos_all,
    means_all,
    width,
    label="All",
    color=color_map["ALL"],
    yerr=yerr_all,
    capsize=4,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.9,
)

# Significantie-boogjes M vs F (per leeftijd, zoals je al had)
y_margin = 0.01

for _, row in signif_df.iterrows():
    age = row["age_bin"]
    if row["p_value"] >= 0.05:
        continue
    center = age_pos[age]
    x_female = center - 0.5 * width      # FEMALE
    x_male   = center + 0.5 * width      # MALE
    y_max = max(row["acc_female"], row["acc_male"])
    y = y_max + 2 * y_margin

    ax.plot(
        [x_female, x_female, x_male, x_male],
        [y, y + y_margin, y + y_margin, y],
        color="black",
        linewidth=0.8,
    )

    ax.text(
        (x_female + x_male) / 2,
        y + 1.8 * y_margin,
        row["sig"],
        ha="center",
        va="bottom",
        fontsize=9,
    )

# # Significantie-boogjes tussen ALL-balken (adjacent age-groepen)
# for _, row in signif_age_overall.iterrows():
#     if row["p_value"] >= 0.05:
#         continue
#     ageA = row["age_A"]
#     ageB = row["age_B"]
#     center_A = age_pos[ageA]
#     center_B = age_pos[ageB]

#     x_A = center_A + 1.5 * width   # ALL positie
#     x_B = center_B + 1.5 * width

#     y_max = max(row["acc_A"], row["acc_B"])
#     y = y_max + 4 * y_margin

#     ax.plot(
#         [x_A, x_A, x_B, x_B],
#         [y, y + y_margin, y + y_margin, y],
#         color="black",
#         linewidth=0.8,
#     )

#     ax.text(
#         (x_A + x_B) / 2,
#         y + 1.8 * y_margin,
#         row["sig"],
#         ha="center",
#         va="bottom",
#         fontsize=9,
#     )

ax.set_xticks(x)
ax.set_xticklabels(age_bins_plot)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Age group")
ax.set_ylim(0.6, 1.0)
ax.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# TIME-PLOT: zelfde idee met paarse ALL-balk + signif tussen ALL
# ------------------------------------------------------------------

results_time =[]
for timing in acc_by_sex_time_ci.index.get_level_values("time_bin").unique():
    FEMALE = acc_by_sex_time_ci.loc[("FEMALE", timing )]
    MALE   = acc_by_sex_time_ci.loc[("MALE", timing)]
    stat, pval = compare_groups(MALE, FEMALE)
    results_time.append({
        "time_bin": timing,
        "acc_male": MALE["accuracy"],
        "acc_female": FEMALE["accuracy"],
        "difference": MALE["accuracy"] - FEMALE["accuracy"],
        "p_value": pval
    })

signif_time_df = pd.DataFrame(results_time)
signif_time_df["sig"] = signif_time_df["p_value"].apply(p_to_star)
print(signif_time_df)

# Significantie tussen ALL (paars) balken per time_bin
results_time_overall = []
time_levels = acc_by_time_ci.index.to_list()

for i in range(len(time_levels) - 1):
    tA = time_levels[i]
    tB = time_levels[i+1]
    rowA = acc_by_time_ci.loc[tA]
    rowB = acc_by_time_ci.loc[tB]
    stat, pval = compare_groups(rowA, rowB)
    results_time_overall.append({
        "time_A": tA,
        "time_B": tB,
        "acc_A": rowA["accuracy"],
        "acc_B": rowB["accuracy"],
        "p_value": pval
    })

signif_time_overall = pd.DataFrame(results_time_overall)
signif_time_overall["sig"] = signif_time_overall["p_value"].apply(p_to_star)
print(signif_time_overall)

df_plot = acc_by_sex_time_ci.reset_index()

fig, ax = plt.subplots(figsize=(7, 4))

time_bins_plot = df_plot["time_bin"].unique()

x = np.arange(len(time_bins_plot))
width = 0.25

time_pos = {t: i for i, t in enumerate(time_bins_plot)}

# Balken voor FEMALE en MALE
for i, gender in enumerate(genders):
    subset = df_plot[df_plot["Gender"] == gender].set_index("time_bin").loc[time_bins_plot]

    means = subset["accuracy"].values
    ci_low = subset["ci_low"].values
    ci_high = subset["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])

    x_pos = x + (i - 0.5) * width

    ax.bar(
        x_pos,
        means,
        width,
        label=gender.capitalize(),
        color=color_map[gender],
        yerr=yerr,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

# Paarse ALL-balken per time_bin
overall_t = acc_by_time_ci.loc[time_bins_plot]
means_all_t = overall_t["accuracy"].values
ci_low_all_t = overall_t["ci_low"].values
ci_high_all_t = overall_t["ci_high"].values
yerr_all_t = np.vstack([means_all_t - ci_low_all_t, ci_high_all_t - means_all_t])

x_pos_all_t = x + 1.5 * width

ax.bar(
    x_pos_all_t,
    means_all_t,
    width,
    label="All",
    color=color_map["ALL"],
    yerr=yerr_all_t,
    capsize=4,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.9,
)

y_margin = 0.01

# M vs F per time_bin
for _, row in signif_time_df.iterrows():
    t = row["time_bin"]
    if row["p_value"] >= 0.05:
        continue
    center = time_pos[t]
    x_female = center - 0.5 * width
    x_male   = center + 0.5 * width
    y_max = max(row["acc_female"], row["acc_male"])
    y = y_max + 2 * y_margin

    ax.plot(
        [x_female, x_female, x_male, x_male],
        [y, y + y_margin, y + y_margin, y],
        color="black",
        linewidth=0.8,
    )

    ax.text(
        (x_female + x_male) / 2,
        y + 1.8 * y_margin,
        row["sig"],
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Significantie tussen ALL-balken (adjacent time-bins)
# for _, row in signif_time_overall.iterrows():
#     if row["p_value"] >= 0.05:
#         continue
#     tA = row["time_A"]
#     tB = row["time_B"]
#     center_A = time_pos[tA]
#     center_B = time_pos[tB]
#     x_A = center_A + 1.5 * width
#     x_B = center_B + 1.5 * width

#     y_max = max(row["acc_A"], row["acc_B"])
#     y = y_max + 4 * y_margin

#     ax.plot(
#         [x_A, x_A, x_B, x_B],
#         [y, y + y_margin, y + y_margin, y],
#         color="black",
#         linewidth=0.8,
#     )

#     ax.text(
#         (x_A + x_B) / 2,
#         y + 1.8 * y_margin,
#         row["sig"],
#         ha="center",
#         va="bottom",
#         fontsize=9,
#     )

ax.set_xticks(x)
ax.set_xticklabels(time_bins_plot)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Time to index ECG")
ax.set_ylim(0.6, 1.0)  
ax.legend()

plt.tight_layout()
plt.show()
