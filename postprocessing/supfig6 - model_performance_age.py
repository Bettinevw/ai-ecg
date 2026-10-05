import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"    
output_file = folder + "prob_age.npz"
output_info = np.load(output_file)
y_pred_test=output_info['y_pred_test']
y_true_test=output_info['y_true_test']
y_test  = np.load(folder+"test_outcome_stratified.npz")["out_age"].astype("int32").reshape(-1)

# Core metrics
mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

print(f"MAE (years):  {mae:.2f}")
print(f"RMSE (years): {rmse:.2f}")
print(f"R2 score:     {r2:.3f}")

# Clinically interpretable metric
within_5y = np.mean(np.abs(y_pred_test - y_test) <= 5.0)
within_10y = np.mean(np.abs(y_pred_test - y_test) <= 10.0)

print(f"Within ±5 years:  {within_5y*100:.1f}%")
print(f"Within ±10 years: {within_10y*100:.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Compute R²
r2 = r2_score(y_test, y_pred_test)

ax[0].scatter(y_test, y_pred_test, alpha=0.5, label=f'R²={np.round(r2,2)}', color='tab:purple')
ax[0].plot([0, 110],[0,110], color='black', linestyle='--')
ax[0].set_title("Regression")
ax[0].set_xlabel("True age [years]")
ax[0].set_ylabel("Predicted age [years]")
ax[0].set_xlim([10,110])
ax[0].set_ylim([10,110])
ax[0].legend()

# Bland-Altman plot
diff = y_pred_test - y_test
md = np.mean(diff)
sd = np.std(diff, axis=0)
ax[1].scatter(y_test, diff, alpha=0.5, color='tab:purple')
ax[1].axhline(md, color='black', linestyle='--')
ax[1].axhline(md + 1.96*sd, color='black', linestyle=':')
ax[1].axhline(md - 1.96*sd, color='black', linestyle=':')
ax[1].set_title("Bland-Altman")
ax[1].set_xlabel("True Age (years)")
ax[1].set_ylabel("Difference (Predicted - True)")
ax[1].set_xlim([10,110])
ax[1].set_ylim([-50,60])

fig.tight_layout()
