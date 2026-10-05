# ============================================================
# Patient-level age subgroup analysis (Route 2)
#   - Descriptive: acc/sens/spec per age bin + 95% bootstrap CI
#   - Inferential: Mann–Whitney U on patient-level p_patient
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG: set your column names
# ----------------------------
PIN_COL        = "PIN"
AGE_INDEX_COL  = "age_at_last_ecg"   # age at index/outcome moment (constant within patient)
YTRUE_COL      = "y_true"            # patient outcome label (0/1), should be constant within patient
PROB_COL       = "y_prob"            # model probability per ECG (0..1)

# Aggregation + threshold
AGG_PROB       = "median"            # "median", "mean", or "max"
# THRESH         = 0.5

# Bootstrapping
N_BOOT         = 2000
SEED           = 42

# Define age bins here
AGE_EDGES      = [17, 60, 70, 80, 200]
AGE_LABELS     = ["18–60","60-70", "70–80", "≥80"]

# ----------------------------
# Helpers
# ----------------------------
def _aggregate_patient_prob(g, prob_col, agg):
    if agg == "median":
        return float(g[prob_col].median())
    if agg == "mean":
        return float(g[prob_col].mean())
    if agg == "max":
        return float(g[prob_col].max())
    raise ValueError("AGG_PROB must be one of: 'median', 'mean', 'max'")

def make_patient_level_df(df):
    """
    One row per patient:
      age_index_years, y_patient, p_patient, yhat_patient, n_ecgs
    """
    df = df.copy()

    # Basic sanity checks
    for c in [PIN_COL, AGE_INDEX_COL, YTRUE_COL, PROB_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    # Check label is constant within patient
    nunique_y = df.groupby(PIN_COL)[YTRUE_COL].nunique()
    n_bad = int((nunique_y > 1).sum())
    if n_bad > 0:
        print(f"Warning: {n_bad} patients have non-constant '{YTRUE_COL}' across ECG rows. Check labels.")

    # Patient-level aggregation
    gb = df.groupby(PIN_COL, sort=False)

    age_patient = gb[AGE_INDEX_COL].first()
    y_patient   = gb[YTRUE_COL].first().astype(int)

    p_patient = gb.apply(lambda g: _aggregate_patient_prob(g, PROB_COL, AGG_PROB))
    n_ecgs    = gb.size().astype(int)

    out = pd.DataFrame({
        PIN_COL: age_patient.index,
        "age_at_last_ecg": age_patient.values.astype(float),
        "y_patient": y_patient.values.astype(int),
        "p_patient": p_patient.values.astype(float),
        "n_ecgs": n_ecgs.values.astype(int),
    })

    out["yhat_patient"] = (out["p_patient"].to_numpy() >= THRESH).astype(int)

    return out

def add_age_bins(df_pat):
    df_pat = df_pat.copy()
    df_pat["age_bin"] = pd.cut(
        df_pat["age_at_last_ecg"],
        bins=AGE_EDGES,
        labels=AGE_LABELS,
        right=False,
        include_lowest=True
    )
    return df_pat

def acc_sens_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return np.nan, np.nan, np.nan
    tn, fp, fn, tp = cm.ravel()

    denom = tp + tn + fp + fn
    acc = (tp + tn) / denom if denom > 0 else np.nan

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(acc), float(sens), float(spec)

def bootstrap_ci_metrics(df_bin, n_boot=N_BOOT, seed=SEED):
    """
    Bootstrap 95% CI on acc/sens/spec within one age bin.
    Resamples patients (rows) with replacement.
    """
    rng = np.random.RandomState(seed)
    idx_all = df_bin.index.to_numpy()

    accs, senss, specs = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        d = df_bin.loc[idx]

        a, s, sp = acc_sens_spec(d["y_patient"].to_numpy(), d["yhat_patient"].to_numpy())
        accs.append(a); senss.append(s); specs.append(sp)

    def ci(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return (np.nan, np.nan)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        "acc_low":  ci(accs)[0],  "acc_high":  ci(accs)[1],
        "sens_low": ci(senss)[0], "sens_high": ci(senss)[1],
        "spec_low": ci(specs)[0], "spec_high": ci(specs)[1],
    }

def descriptive_table_with_ci(df_pat):
    """
    Returns a table with acc/sens/spec per age bin + 95% bootstrap CI (descriptive).
    """
    rows = []
    for b in df_pat["age_bin"].cat.categories:
        sub = df_pat[df_pat["age_bin"] == b]
        n = int(len(sub))
        if n == 0:
            rows.append({
                "age_bin": str(b), "n_patients": 0,
                "acc": np.nan, "acc_low": np.nan, "acc_high": np.nan,
                "sens": np.nan, "sens_low": np.nan, "sens_high": np.nan,
                "spec": np.nan, "spec_low": np.nan, "spec_high": np.nan,
            })
            continue

        acc, sens, spec = acc_sens_spec(sub["y_patient"].to_numpy(), sub["yhat_patient"].to_numpy())
        ci = bootstrap_ci_metrics(sub)

        rows.append({
            "age_bin": str(b), "n_patients": n,
            "acc": acc, **{k: ci[k] for k in ["acc_low", "acc_high"]},
            "sens": sens, **{k: ci[k] for k in ["sens_low", "sens_high"]},
            "spec": spec, **{k: ci[k] for k in ["spec_low", "spec_high"]},
        })

    return pd.DataFrame(rows)

def mann_whitney_pairwise(df_pat, adjust="holm"):
    """
    Pairwise Mann–Whitney U tests on p_patient between age bins (inferential).
    Returns a DataFrame with raw and adjusted p-values.
    """
    bins = list(df_pat["age_bin"].cat.categories)
    pairs = list(itertools.combinations(bins, 2))

    pvals = []
    ns = []
    for a, b in pairs:
        x = df_pat.loc[df_pat["age_bin"] == a, "p_patient"].dropna()
        y = df_pat.loc[df_pat["age_bin"] == b, "p_patient"].dropna()

        ns.append((int(len(x)), int(len(y))))

        # If one of the groups is empty, return NaN
        if len(x) == 0 or len(y) == 0:
            pvals.append(np.nan)
            continue

        _, p = mannwhitneyu(x, y, alternative="two-sided")
        pvals.append(float(p))

    # Multiple-testing correction (ignore NaNs for adjustment)
    pvals_arr = np.array(pvals, dtype=float)
    valid = np.isfinite(pvals_arr)

    p_adj = np.full_like(pvals_arr, np.nan, dtype=float)
    if valid.sum() > 0:
        rej, pvals_corr, _, _ = multipletests(pvals_arr[valid], method=adjust)
        p_adj[valid] = pvals_corr

    out = pd.DataFrame({
        "age_bin_a": [str(a) for a, b in pairs],
        "age_bin_b": [str(b) for a, b in pairs],
        "n_a": [na for (na, nb) in ns], # number a
        "n_b": [nb for (na, nb) in ns], # number b
        "p_raw": pvals_arr, 
        f"p_{adjust}": p_adj
    }).sort_values(f"p_{adjust}")

    return out

def p_to_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ----------------------------
# RUN (copy-paste and execute)
# ----------------------------
# df should be your ECG-level dataframe with columns:
# PIN, age_index_years, y_true (patient outcome), y_prob (model prob per ECG)

# 1) Patient-level aggregation (Route 2)
# df_pat = make_patient_level_df(df)

# 2) Add age bins
# df_pat = add_age_bins(df_pat)

# 3) DESCRIPTIVE: acc/sens/spec per age bin + 95% CI
# desc = descriptive_table_with_ci(df_pat)
# print(desc)

# 4) INFERENTIAL: Mann–Whitney U on p_patient (pairwise + Holm correction)
# mw = mann_whitney_pairwise(df_pat, adjust="holm")
# print(mw)

# Optional: if you only want one global comparison (young vs old),
# filter df_pat to two bins and run mannwhitneyu directly.



# -------------------------
# Example usage
# -------------------------
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = 'stratified'
test_file = folder + "dataset_test_" + name + ".xlsx"
df      = pd.read_excel(test_file)
X_test  = np.load(folder+"test_dataset_" + name + ".npy")["ecg"].astype("float32")
y_test  = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob = np.load(output_file)["probs_test"].astype("float32")
THRESH = np.load(output_file)["thr"].astype("float32")

df["y_prob"] = y_prob 

days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3])*days_per_year

labels = ["≤1w", "1w–1m", "1–3m", "3–6m", "6–12m", "1-2y", "2-3y"]
df["time_bin"] = pd.cut(df['days_to_index'], bins=time_bins, labels=labels, right=False)
df["time_bin"] = pd.Categorical(df["time_bin"],
                                categories=labels[::-1],
                                ordered=True)

df["y_pred"] = (df["y_prob"] >= THRESH).astype(int)
df["y_true"] = y_test[df.index]

# 1) patient-level aggregation
df_pat = make_patient_level_df(df)

# 2) Add age bins
df_pat = add_age_bins(df_pat)

# 3) DESCRIPTIVE: acc/sens/spec per age bin + 95% CI
desc = descriptive_table_with_ci(df_pat)
print(desc)

# 4) INFERENTIAL: Mann–Whitney U on p_patient (pairwise + Holm correction)
mw = mann_whitney_pairwise(df_pat, adjust="holm")
print(mw)

### plot

fig, ax = plt.subplots(1,3, figsize=(12, 4))
x = np.arange(len(AGE_LABELS))
width = 0.33  # 
time_pos = {time: i for i, time in enumerate(AGE_LABELS)}

for i_label in range(len(AGE_LABELS)):
    stats = desc[desc['age_bin'] == AGE_LABELS[i_label]]
    
    ax[0].bar(x[i_label],
        stats['acc'],
        width,
        color='tab:purple',
        yerr=np.vstack([stats['acc'] - stats['acc_low'], stats['acc_high'] - stats['acc']]),
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9)
    ax[1].bar(x[i_label],
        stats['sens'],
        width,
        color='tab:purple',
        yerr=np.vstack([stats['sens'] - stats['sens_low'], stats['sens_high'] - stats['sens']]),
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9)
    ax[2].bar(x[i_label],
        stats['spec'],
        width,
        color='tab:purple',
        yerr=np.vstack([stats['spec'] - stats['spec_low'], stats['spec_high'] - stats['spec']]),
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9)


ax[0].set_xticks(x)
ax[0].set_xticklabels(AGE_LABELS)
ax[0].set_ylabel("Accuracy")
ax[0].set_xlabel("Age bin [year]")
ax[0].set_ylim(0, 1.0)
ax[0].grid('on')

ax[1].set_xticks(x)
ax[1].set_xticklabels(AGE_LABELS)
ax[1].set_ylabel("Sensitivity")
ax[1].set_xlabel("Age bin [year]")
ax[1].set_ylim(0, 1.0)
ax[1].grid('on')

ax[2].set_xticks(x)
ax[2].set_xticklabels(AGE_LABELS)
ax[2].set_ylabel("Specificity")
ax[2].set_xlabel("Age bin [year]")
ax[2].set_ylim(0, 1.0)
ax[2].grid('on')

plt.tight_layout()
plt.show()
