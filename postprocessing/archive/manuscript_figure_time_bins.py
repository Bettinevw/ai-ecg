import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import multipletests

# -----------------------------
# 1) Helpers: metrics per bin
# -----------------------------
def _safe_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return None
    return cm.ravel()  # tn, fp, fn, tp

def compute_metrics_per_bin(df, bin_ids):
    out = {}
    for b in bin_ids:
        d = df[df[BIN_COL] == b]
        if len(d) == 0:
            out[b] = {"acc": np.nan, "sens": np.nan, "spec": np.nan}
            continue

        y_true = d[YTRUE_COL].to_numpy()
        y_pred = d[YPRED_COL].to_numpy()

        tmp = _safe_confusion(y_true, y_pred)
        if tmp is None:
            out[b] = {"acc": np.nan, "sens": np.nan, "spec": np.nan}
            continue

        tn, fp, fn, tp = tmp
        denom = tn + fp + fn + tp
        acc = (tp + tn) / denom if denom > 0 else np.nan

        denom_sens = tp + fn
        sens = tp / denom_sens if denom_sens > 0 else np.nan

        denom_spec = tn + fp
        spec = tn / denom_spec if denom_spec > 0 else np.nan

        out[b] = {"acc": float(acc), "sens": float(sens), "spec": float(spec)}
    return out

def ci_from_boot(arr, alpha=0.05):
    arr = np.array(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    lo = np.percentile(arr, 100*(alpha/2))
    hi = np.percentile(arr, 100*(1 - alpha/2))
    return float(lo), float(hi)

def p_two_sided_from_boot(arr):
    """
    Two-sided p-value for H0: statistic == 0 based on bootstrap distribution.
    """
    arr = np.array(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
    return float(min(p, 1.0))

# -----------------------------
# 2) Patient-level cluster bootstrap
# -----------------------------
def cluster_bootstrap_metrics(df, bin_ids, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    pins = df[PIN_COL].unique()

    boot_metrics = {b: {"acc": [], "sens": [], "spec": []} for b in bin_ids}

    for _ in range(n_boot):
        sampled = rng.choice(pins, size=len(pins), replace=True)

        parts = [df[df[PIN_COL] == p] for p in sampled]  # keeps multiplicities
        df_b = pd.concat(parts, axis=0, ignore_index=True)

        m = compute_metrics_per_bin(df_b, bin_ids)

        for b in bin_ids:
            for k in ("acc", "sens", "spec"):
                boot_metrics[b][k].append(m[b][k])

    for b in bin_ids:
        for k in ("acc", "sens", "spec"):
            boot_metrics[b][k] = np.array(boot_metrics[b][k], dtype=float)

    return boot_metrics

def cluster_bootstrap_pairwise_diffs(df, bin_ids, n_boot=2000, seed=42):
    """
    Returns bootstrap distributions of pairwise differences between bins:
      diffs[(b1, b2)][metric] = array of (metric(b2) - metric(b1)) across bootstraps
    """
    rng = np.random.RandomState(seed)
    pins = df[PIN_COL].unique()

    pairs = []
    for i in range(len(bin_ids)):
        for j in range(i+1, len(bin_ids)):
            pairs.append((bin_ids[i], bin_ids[j]))

    diffs = {pair: {"acc": [], "sens": [], "spec": []} for pair in pairs}

    for _ in range(n_boot):
        sampled = rng.choice(pins, size=len(pins), replace=True)
        parts = [df[df[PIN_COL] == p] for p in sampled]
        df_b = pd.concat(parts, axis=0, ignore_index=True)

        m = compute_metrics_per_bin(df_b, bin_ids)

        for (b1, b2) in pairs:
            for k in ("acc", "sens", "spec"):
                diffs[(b1, b2)][k].append(m[b2][k] - m[b1][k])  # b2 - b1

    for pair in pairs:
        for k in ("acc", "sens", "spec"):
            diffs[pair][k] = np.array(diffs[pair][k], dtype=float)

    return diffs

# -----------------------------
# 3) Summaries: per-bin CI + pairwise significance
# -----------------------------
def summarize_bins(df, bin_ids, boot_metrics):
    point = compute_metrics_per_bin(df, bin_ids)

    rows = []
    for b in bin_ids:
        for k in ("acc", "sens", "spec"):
            lo, hi = ci_from_boot(boot_metrics[b][k], alpha=0.05)
            rows.append({
                "metric": k,
                "bin_label": b,
                "point": point[b][k],
                "ci_low": lo,
                "ci_high": hi
            })
    return pd.DataFrame(rows)

def summarize_pairwise(diffs, adjust="holm"):
    """
    Produces a table with point delta (from original data not included here),
    bootstrap CI, bootstrap p-value, and Holm-adjusted p-value per metric.
    """
    rows = []
    for (b1, b2), d in diffs.items():
        for k in ("acc", "sens", "spec"):
            arr = d[k]
            lo, hi = ci_from_boot(arr, alpha=0.05)
            p = p_two_sided_from_boot(arr)
            rows.append({
                "bin_a": b1,
                "bin_b": b2,
                "metric": k,
                "delta_b_minus_a": np.nanmean(arr),  # bootstrap mean (fine as summary of delta dist)
                "ci_low": lo,
                "ci_high": hi,
                "p_raw": p
            })

    out = pd.DataFrame(rows)

    # Holm correction per metric
    out["p_holm"] = np.nan
    for k in ("acc", "sens", "spec"):
        mask = (out["metric"] == k) & np.isfinite(out["p_raw"].to_numpy())
        pvals = out.loc[mask, "p_raw"].to_numpy(dtype=float)
        if len(pvals) > 0:
            _, p_corr, _, _ = multipletests(pvals, method=adjust)
            out.loc[mask, "p_holm"] = p_corr

    return out

# -----------------------------
# 4) RUN: your data prep stays the same up to df["y_true"], df["y_pred"], df["time_bin"]
# -----------------------------
# CONFIG (your names)
PIN_COL   = "PIN"
BIN_COL   = "time_bin"
YTRUE_COL = "y_true"
YPRED_COL = "y_pred"
PROB_COL  = None
# THRESH    = 0.5

N_BOOT = 2000
SEED   = 42

# Your labels (choose order for analysis; here we use the plotted order)
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
# If you want reverse order (as in your plot):
bin_ids = labels[::-1]
# or if you want exactly the categorical order you set:
# bin_ids = list(df[BIN_COL].cat.categories)

# Bootstrap per-bin metrics + pairwise diffs
boot_metrics = cluster_bootstrap_metrics(df, bin_ids, n_boot=N_BOOT, seed=SEED)
summary_bins = summarize_bins(df, bin_ids, boot_metrics)

diffs = cluster_bootstrap_pairwise_diffs(df, bin_ids, n_boot=N_BOOT, seed=SEED)
summary_pairs = summarize_pairwise(diffs, adjust="holm")  # includes p_raw and p_holm per metric

# Save
np.savez(folder+'figure_time_bins',
         summary_bins=summary_bins.to_records(index=False),
         summary_pairs=summary_pairs.to_records(index=False))

# -----------------------------
# 5) Plot bars with CI (your style), no slopes
# -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
x = np.arange(len(bin_ids))
width = 0.33
time_pos = {lab: i for i, lab in enumerate(bin_ids)}

for lab in bin_ids:
    acc_ci  = summary_bins[(summary_bins["metric"] == "acc")  & (summary_bins["bin_label"] == lab)].iloc[0]
    sens_ci = summary_bins[(summary_bins["metric"] == "sens") & (summary_bins["bin_label"] == lab)].iloc[0]
    spec_ci = summary_bins[(summary_bins["metric"] == "spec") & (summary_bins["bin_label"] == lab)].iloc[0]

    ax[0].bar(time_pos[lab], acc_ci["point"], width,
              yerr=np.array([[acc_ci["point"] - acc_ci["ci_low"]],
                             [acc_ci["ci_high"] - acc_ci["point"]]]),
              capsize=4, edgecolor="black", linewidth=0.5, alpha=0.9)

    ax[1].bar(time_pos[lab], sens_ci["point"], width,
              yerr=np.array([[sens_ci["point"] - sens_ci["ci_low"]],
                             [sens_ci["ci_high"] - sens_ci["point"]]]),
              capsize=4, edgecolor="black", linewidth=0.5, alpha=0.9)

    ax[2].bar(time_pos[lab], spec_ci["point"], width,
              yerr=np.array([[spec_ci["point"] - spec_ci["ci_low"]],
                             [spec_ci["ci_high"] - spec_ci["point"]]]),
              capsize=4, edgecolor="black", linewidth=0.5, alpha=0.9)

for i, title in enumerate(["Accuracy", "Sensitivity", "Specificity"]):
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(bin_ids)
    ax[i].set_ylabel(title)
    ax[i].set_xlabel("Time to index ECG")
    ax[i].set_ylim(0, 1.0)
    ax[i].grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# 6) Optional: draw significance brackets from summary_pairs
#     Keep it readable: often use ONE reference bin vs others, or adjacent only.
# -----------------------------
def p_to_star(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def add_brackets(ax, summary_pairs, metric="acc", p_col="p_holm",
                 only_adjacent=True, y_margin=0.01):
    # filter for this metric
    sp = summary_pairs[summary_pairs["metric"] == metric].copy()

    # optionally keep only adjacent comparisons (much less clutter)
    if only_adjacent:
        idx = {lab: i for i, lab in enumerate(bin_ids)}
        sp = sp[sp.apply(lambda r: abs(idx[r["bin_b"]] - idx[r["bin_a"]]) == 1, axis=1)]

    # sort by bracket width
    idx = {lab: i for i, lab in enumerate(bin_ids)}
    sp = sp.sort_values(by=["bin_a", "bin_b"])

    k = 0
    for _, r in sp.iterrows():
        p = r[p_col]
        if not np.isfinite(p) or p >= 0.05:
            continue

        a = r["bin_a"]; b = r["bin_b"]
        x_left, x_right = idx[a], idx[b]
        if x_left > x_right:
            x_left, x_right = x_right, x_left

        # get bar heights from summary_bins
        v_left  = summary_bins[(summary_bins["metric"] == metric) & (summary_bins["bin_label"] == a)]["point"].iloc[0]
        v_right = summary_bins[(summary_bins["metric"] == metric) & (summary_bins["bin_label"] == b)]["point"].iloc[0]
        y_max = max(v_left, v_right)

        y = y_max + (2 + 2*k)*y_margin
        ax.plot([x_left, x_left, x_right, x_right],
                [y, y + y_margin, y + y_margin, y],
                color="black", linewidth=0.8)
        ax.text((x_left + x_right)/2, y + 1.8*y_margin, p_to_star(p),
                ha="center", va="bottom", fontsize=9)
        k += 1

# Example: only adjacent significant brackets on accuracy axis
# add_brackets(ax[0], summary_pairs, metric="acc", p_col="p_holm", only_adjacent=True, y_margin=0.01)
# plt.show()
