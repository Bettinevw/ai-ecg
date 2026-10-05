import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

main_folder   = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"
AF_file       = main_folder + "prepared_AF_group.xlsx"
control_file  = main_folder + "prepared_control_group.xlsx"

# 1) Load only needed columns to speed up
usecols_af   = ["PIN","Gender","PatientAge","Group","days_to_index", "age_at_last_ecg"]
usecols_ctrl = ["PIN","Gender","PatientAge","Group","days_to_index", "age_at_last_ecg"]
df_a    = pd.read_excel(AF_file,   usecols=usecols_af)
df_c    = pd.read_excel(control_file, usecols=usecols_ctrl)
print('Af and Ctrl group loaded')

df = pd.concat([df_c, df_a], ignore_index=True)
df = df[df["days_to_index"] != 0].copy() # Exclude index ECG

df = pd.concat([df_c, df_a], ignore_index=True)
df_age_unique = df.groupby("PIN", as_index=False).agg({
                            "age_at_last_ecg": "first",
                            "Gender": "first",
                            "Group": "first"})
# bins = np.array([17, 55, 65, 75, 130])
bins= np.arange(0, 130, 5)
# --- Age ---
fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
subset_cM = df_age_unique[(df_age_unique["Group"] =="Control") & (df_age_unique["Gender"] == "MALE")]
subset_cF = df_age_unique[(df_age_unique["Group"] == "Control") & (df_age_unique["Gender"] == "FEMALE")]

subset_aM = df_age_unique[(df_age_unique["Group"] =="AF") & (df_age_unique["Gender"] == "MALE")]
subset_aF = df_age_unique[(df_age_unique["Group"] == "AF") & (df_age_unique["Gender"] == "FEMALE")]
    
axs[0].hist(subset_cM["age_at_last_ecg"], bins=bins, alpha=0.4,
          label="ctrl-Male",
          edgecolor="black", color = 'tab:blue')
axs[0].hist(subset_aM["age_at_last_ecg"], bins=bins, alpha=0.4,
          label="AF-Male",
          edgecolor="black", color = 'tab:orange')

axs[1].hist(subset_cF["age_at_last_ecg"], bins=bins, alpha=0.4,
          label="ctrl-Female",
          edgecolor="black", color = 'tab:blue')
axs[1].hist(subset_aF["age_at_last_ecg"], bins=bins, alpha=0.4,
          label="AF-Female",
          edgecolor="black", color = 'tab:orange')
        

axs[0].set_xlabel("Patient Age at index ECG (years)")
axs[0].set_ylabel("Count")
# plt.title("Age distribution: AF vs Control")
axs[0].legend()
axs[1].legend()
# plt.show()

age_bins = np.array([17, 60, 70, 80, 130])

# --- Age ---
fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
    
axs[0].hist(subset_cM["age_at_last_ecg"], bins=age_bins, alpha=0.4,
          label="ctrl-Male",
          edgecolor="black", color = 'tab:blue')
axs[0].hist(subset_aM["age_at_last_ecg"], bins=age_bins, alpha=0.4,
          label="AF-Male",
          edgecolor="black", color = 'tab:orange')

axs[1].hist(subset_cF["age_at_last_ecg"], bins=age_bins, alpha=0.4,
          label="ctrl-Female",
          edgecolor="black", color = 'tab:blue')
axs[1].hist(subset_aF["age_at_last_ecg"], bins=age_bins, alpha=0.4,
          label="AF-Female",
          edgecolor="black", color = 'tab:orange')
        

axs[0].set_xlabel("Patient Age (years)")
axs[0].set_ylabel("Count")
# plt.title("Age distribution: AF vs Control")
axs[0].legend()
axs[1].legend()


subset_cM = df[(df["Group"] =="Control") & (df["Gender"] == "MALE")]
subset_cF = df[(df["Group"] == "Control") & (df["Gender"] == "FEMALE")]

subset_aM = df[(df["Group"] =="AF") & (df["Gender"] == "MALE")]
subset_aF = df[(df["Group"] == "AF") & (df["Gender"] == "FEMALE")]

# --- Timing ---
# unify possible column names

time_bins = np.arange(0, 10000, 60) #[0, 7, 30, 90, 180, 365, 730, 1095, 1825, 2000]
fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
axs[0].hist(subset_cM["days_to_index"], alpha=0.4,
          label="ctrl-Male",
          edgecolor="black", color = 'tab:blue')
axs[0].hist(subset_aM["days_to_index"], alpha=0.4,
          label="AF-Male",
          edgecolor="black", color = 'tab:orange')

axs[1].hist(subset_cF["days_to_index"], alpha=0.4,
          label="ctrl-Female",
          edgecolor="black", color = 'tab:blue')
axs[1].hist(subset_aF["days_to_index"], alpha=0.4,
          label="AF-Female",
          edgecolor="black", color = 'tab:orange')
axs[0].set_xlabel("Days between ECGs / days to index ECG")
axs[0].set_ylabel("Count")
# plt.title("Timing distribution: AF vs Control")
# plt.legend()
# plt.show()

days_per_year = 365.25
time_bins = np.array([0, 7/days_per_year, 30/days_per_year, 0.25, 0.5, 1, 2, 3, 4, 5])*days_per_year
fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10, 4))
axs[0].hist(subset_cM["days_to_index"], bins=time_bins, alpha=0.4,
          label="ctrl-Male",
          edgecolor="black", color = 'tab:blue')
axs[0].hist(subset_aM["days_to_index"], bins=time_bins, alpha=0.4,
          label="AF-Male",
          edgecolor="black", color = 'tab:orange')

axs[1].hist(subset_cF["days_to_index"], bins=time_bins, alpha=0.4,
          label="ctrl-Female",
          edgecolor="black", color = 'tab:blue')
axs[1].hist(subset_aF["days_to_index"], bins=time_bins, alpha=0.4,
          label="AF-Female",
          edgecolor="black", color = 'tab:orange')
axs[0].set_xlabel("Days between ECGs / days to index ECG")
axs[0].set_ylabel("Count")

def make_heatmap(df, ax, title, age_bins, time_bins, cmap="viridis"):
    """Build a 2D histogram (counts) over Age x days_to_index and draw a heatmap."""
    # Bin the data
    age_cat  = pd.cut(df["age_at_last_ecg"], bins=age_bins, right=False)
    time_cat = pd.cut(df["days_to_index"], bins=time_bins, right=False)

    # Crosstab to counts
    H = pd.crosstab(time_cat, age_cat)  # rows=y (time), cols=x (age)

    # Ensure full grid coverage including empty bins
    H = H.reindex(index=pd.IntervalIndex(pd.cut(time_bins[:-1], time_bins, right=False).categories, name=time_cat.name),
                  columns=pd.IntervalIndex(pd.cut(age_bins[:-1],  age_bins,  right=False).categories, name=age_cat.name),
                  fill_value=0)

    # Convert to numpy array for imshow
    A = H.to_numpy()
    
    print("min nr:", np.min(A))

    # Plot heatmap
    im = ax.imshow(A, aspect="auto", origin="lower", cmap=cmap)

    # Build tick labels from bin intervals
    xlabels = [f"[{iv.left},{iv.right})" for iv in H.columns]
    ylabels = [f"[{iv.left},{iv.right})" for iv in H.index]
    
    # Voeg nummers toe in de cellen
    A = H.to_numpy()
    for i in range(A.shape[0]):        # over y
        for j in range(A.shape[1]):    # over x
            val = A[i, j]
            if val > 0:                # toon alleen niet-lege cellen
                ax.text(j, i, str(int(val)),
                        ha="center", va="center",
                        color="white" if val > A.max()/2 else "black",
                        fontsize=8)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Age bin [years]")
    ax.set_ylabel("days_to_index bin [days]")
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("# ECGs")

    return H  # optional: return the table of counts for further checks

fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
H_cm  = make_heatmap(subset_cM, axs[0,0], "ctrl M", age_bins, time_bins)
H_cf  = make_heatmap(subset_cF, axs[0,1], "ctrl F", age_bins, time_bins)
H_am  = make_heatmap(subset_aM, axs[1,0], "af M", age_bins, time_bins)
H_af  = make_heatmap(subset_aF, axs[1,1], "af F", age_bins, time_bins)
plt.show()
