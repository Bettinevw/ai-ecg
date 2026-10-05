# ============================================================
# Patient-level age x sex subgroup analysis (Route 2)
#   - Descriptive: acc/sens/spec per (age bin, sex) + 95% bootstrap CI
#   - Inferential: per age bin, Mann–Whitney U on p_patient: male vs female
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG: set your column names
# ----------------------------
PIN_COL        = "PIN"
AGE_INDEX_COL  = "age_at_last_ecg"     # constant within patient
SEX_COL        = "Gender"                # e.g., 'M'/'F' or 0/1; see normalize_sex()
YTRUE_COL      = "y_true"             # 0/1, constant within patient
PROB_COL       = "y_prob"             # per-ECG probability

AGG_PROB       = "median"             # "median", "mean", "max"
N_BOOT         = 2000
SEED           = 42

AGE_EDGES      = [17, 60, 70, 80, 200]
AGE_LABELS     = ["18–60", "60–70", "70–80", "≥80"]

# Plot settings
SEX_ORDER      = ["FEMALE", "MALE"]           # order on plot (and used in computations)
SEX_LABELS     = {"FEMALE": "Female", "MALE": "Male"}  # for legend text

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

def make_patient_level_df(df, thresh):
    """
    One row per patient:
      age, sex, y_patient, p_patient, yhat_patient, n_ecgs
    """
    df = df.copy()

    for c in [PIN_COL, AGE_INDEX_COL, SEX_COL, YTRUE_COL, PROB_COL]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    # Warn if label not constant per patient
    nunique_y = df.groupby(PIN_COL)[YTRUE_COL].nunique()
    n_bad = int((nunique_y > 1).sum())
    if n_bad > 0:
        print(f"Warning: {n_bad} patients have non-constant '{YTRUE_COL}' across ECG rows. Check labels.")

    gb = df.groupby(PIN_COL, sort=False)

    age_patient = gb[AGE_INDEX_COL].first().astype(float)
    sex_patient = gb[SEX_COL].first()
    y_patient   = gb[YTRUE_COL].first().astype(int)

    p_patient = gb.apply(lambda g: _aggregate_patient_prob(g, PROB_COL, AGG_PROB))
    n_ecgs    = gb.size().astype(int)

    out = pd.DataFrame({
        PIN_COL: age_patient.index,
        "age_at_last_ecg": age_patient.values,
        "Gender": sex_patient.values,
        "y_patient": y_patient.values,
        "p_patient": p_patient.values.astype(float),
        "n_ecgs": n_ecgs.values,
    })

    out["yhat_patient"] = (out["p_patient"].to_numpy() >= float(thresh)).astype(int)
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

def bootstrap_ci_metrics(df_sub, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    idx_all = df_sub.index.to_numpy()
    accs, senss, specs = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        d = df_sub.loc[idx]
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

def descriptive_table_with_ci_by_age_sex(df_pat):
    """
    One row per (age_bin, sex) with point estimates + 95% bootstrap CI.
    """
    rows = []
    for b in df_pat["age_bin"].cat.categories:
        for sx in SEX_ORDER:
            sub = df_pat[(df_pat["age_bin"] == b) & (df_pat["Gender"] == sx)]
            n = int(len(sub))
            if n == 0:
                rows.append({
                    "age_bin": str(b), "Gender": sx, "n_patients": 0,
                    "acc": np.nan, "acc_low": np.nan, "acc_high": np.nan,
                    "sens": np.nan, "sens_low": np.nan, "sens_high": np.nan,
                    "spec": np.nan, "spec_low": np.nan, "spec_high": np.nan,
                })
                continue

            acc, sens, spec = acc_sens_spec(sub["y_patient"].to_numpy(), sub["yhat_patient"].to_numpy())
            ci = bootstrap_ci_metrics(sub)

            rows.append({
                "age_bin": str(b), "Gender": sx, "n_patients": n,
                "acc": acc, **{k: ci[k] for k in ["acc_low", "acc_high"]},
                "sens": sens, **{k: ci[k] for k in ["sens_low", "sens_high"]},
                "spec": spec, **{k: ci[k] for k in ["spec_low", "spec_high"]},
            })
    return pd.DataFrame(rows)

def mw_male_vs_female_per_agebin(df_pat, adjust="holm"):
    """
    For each age bin: Mann–Whitney U test on p_patient comparing M vs F.
    Returns p_raw and adjusted p-values across age bins.
    """
    rows = []
    for b in df_pat["age_bin"].cat.categories:
        xF = df_pat[(df_pat["age_bin"] == b) & (df_pat["Gender"] == "FEMALE")]["p_patient"].dropna()
        xM = df_pat[(df_pat["age_bin"] == b) & (df_pat["Gender"] == "MALE")]["p_patient"].dropna()

        nF, nM = int(len(xF)), int(len(xM))
        if nF == 0 or nM == 0:
            p = np.nan
        else:
            _, p = mannwhitneyu(xF, xM, alternative="two-sided")
            p = float(p)

        rows.append({
            "age_bin": str(b),
            "n_F": nF,
            "n_M": nM,
            "p_raw": p
        })

    out = pd.DataFrame(rows)
    pvals = out["p_raw"].to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    out[f"p_{adjust}"] = np.nan
    if valid.sum() > 0:
        _, p_corr, _, _ = multipletests(pvals[valid], method=adjust)
        out.loc[valid, f"p_{adjust}"] = p_corr
    return out

def p_to_star(p):
    if not np.isfinite(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ----------------------------
# RUN
# ----------------------------
folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
name = "stratified"
test_file = folder + "dataset_test_" + name + ".xlsx"
df = pd.read_excel(test_file)

y_test = np.load(folder+"test_outcome_" + name + ".npz")["outcome"].astype("int32").reshape(-1)
output_file = folder + "prob_"+name+".npz"
y_prob = np.load(output_file)["probs_test"].astype("float32")
THRESH = float(np.load(output_file)["thr"].astype("float32"))

df["y_prob"] = y_prob
df["y_true"] = y_test[df.index]

# df must contain a sex column; if it's called differently, set SEX_COL above.
# df[SEX_COL] should exist here.

# Patient-level aggregation
df_pat = make_patient_level_df(df, thresh=THRESH)
df_pat = add_age_bins(df_pat)

# Descriptive table (by age x sex)
desc = descriptive_table_with_ci_by_age_sex(df_pat)
print(desc)

# Inferential: per age bin, M vs F MW test on p_patient (+ Holm correction across age bins)
mw_sex = mw_male_vs_female_per_agebin(df_pat, adjust="holm")
print(mw_sex)

# ----------------------------
# Plot: 3 panels, grouped bars (Female vs Male) per age bin, with CI + brackets
# ----------------------------
colors = {"FEMALE": "gold", "MALE": "tab:blue"}   # <-- requested colors
width = 0.35
offset = {"FEMALE": -width/2, "MALE": width/2}

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
x = np.arange(len(AGE_LABELS))

def add_sex_bracket(ax_, x_center, y_left, y_right, p, y_margin=0.015, linew=0.9):
    """
    Draws a bracket connecting the two bars (female/male) within one age bin.
    x_center: bin center position (integer)
    y_left/y_right: heights of the two bars for this metric (scalars)
    p: p-value to annotate as stars
    """
    if not np.isfinite(p) or p >= 0.05:
        return

    # bar x positions
    x_f = x_center + offset["FEMALE"]
    x_m = x_center + offset["MALE"]

    # y position just above the taller bar
    y_top = float(max(y_left, y_right))
    y = y_top + y_margin

    # bracket
    ax_.plot([x_f, x_f, x_m, x_m],
             [y,   y + y_margin, y + y_margin, y],
             color="black", linewidth=linew)

    ax_.text((x_f + x_m)/2, y + 1.6*y_margin, p_to_star(p),
             ha="center", va="bottom", fontsize=10)

# ---- bars + CIs
for i, metric in enumerate(["acc", "sens", "spec"]):
    for sx in SEX_ORDER:
        sub = desc[desc["Gender"] == sx].set_index("age_bin").reindex(AGE_LABELS)

        y = sub[metric].to_numpy(dtype=float)
        ylo = sub[f"{metric}_low"].to_numpy(dtype=float)
        yhi = sub[f"{metric}_high"].to_numpy(dtype=float)

        yerr = np.vstack([y - ylo, yhi - y])

        ax[i].bar(
            x + offset[sx],
            y,
            width=width,
            yerr=yerr,
            capsize=4,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            color=colors[sx],                 # <-- color set here
            label=SEX_LABELS.get(sx, sx) if i == 0 else None  # only add legend once
        )

    ax[i].set_ylabel({"acc":"Accuracy","sens":"Sensitivity","spec":"Specificity"}[metric])
    ax[i].set_ylim(0, 1.0)
    ax[i].grid(True)

ax[0].legend(loc="upper right")

for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(AGE_LABELS)
    a.set_xlabel("Age bin [years]")

plt.tight_layout()
plt.show()
