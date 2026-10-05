import os, sys
import numpy as np
from tensorflow import keras

# Add preprocessing directory to work directory
dir_path        = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/model/')

from model_keras import ResBlock
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve

folder = "G:/diva1/Research/MUSE ECGs/PROJECTS/Bettine/Data/"

name='stratified'
# name = 'prevalence'

X_val   = np.load(folder+"val_dataset_" +name+".npy")["ecg"].astype("float32")
X_test  = np.load(folder+"test_dataset_" +name+".npy")["ecg"].astype("float32")
X_train = np.load(folder+"train_dataset_" +name+".npy")["ecg"].astype("float32")

model_type = 'age' #'AF' # 'age', 'sex'

if model_type == 'AF':
    
    y_test  = np.load(folder+"test_outcome_" +name+".npz")["outcome"].astype("int32").reshape(-1)
    y_val   = np.load(folder+"val_outcome_" +name+".npz")["outcome"].astype("int32").reshape(-1)
    y_train = np.load(folder+"train_outcome_" +name+".npz")["outcome"].astype("int32").reshape(-1)
    
    best_model_file = folder + "model_" +name+".h5"
    output_file = folder + "prob_" +name+".npz"
    
    best_model = keras.models.load_model(best_model_file,
                                         custom_objects={'ResBlock': ResBlock})

    vals = best_model.evaluate(X_test, y_test, verbose=0)
    print(dict(zip(best_model.metrics_names, [float(v) for v in vals])))

    probs_val  = best_model.predict(X_val,  verbose=0)[:, 0]
    probs_test = best_model.predict(X_test, verbose=0)[:, 0]
        
    prec, rec, thr = precision_recall_curve(y_val, probs_val)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    t_best = float(thr[np.nanargmax(f1)])

    y_hat = (probs_test >= t_best).astype("int32")
    print("ROC-AUC:", roc_auc_score(y_test, probs_test))
    print(classification_report(y_test, y_hat, digits=4))
    print(confusion_matrix(y_test, y_hat))

    np.savez(output_file, probs_val=probs_val, probs_test=probs_test, 
             thr=t_best, y_hat=y_hat)
      

elif model_type == 'age':
    
    y_test  = np.load(folder+"test_outcome_" +name+".npz")["out_age"].astype("int32").reshape(-1)
    y_val   = np.load(folder+"val_outcome_" +name+".npz")["out_age"].astype("int32").reshape(-1)
    y_train = np.load(folder+"train_outcome_" +name+".npz")["out_age"].astype("int32").reshape(-1)
    
    best_model_file = folder + "model_age_stratified.h5"
    output_file = folder + "prob_age.npz"
    
    # testen
    best_model = keras.models.load_model(best_model_file,
                                         custom_objects={'ResBlock': ResBlock})
    
    # Predict continuous values
    y_pred_test = best_model.predict(X_test, verbose=0).squeeze()
    
    
    # Save outputs
    np.savez(
        output_file,
        y_pred_test=y_pred_test,
        y_true_test=y_test
    )
    

    
    
elif model_type == 'sex':
    
    y_test  = np.load(folder+"test_outcome_" +name+".npz")["out_sex"].astype("int32").reshape(-1)
    y_val   = np.load(folder+"val_outcome_" +name+".npz")["out_sex"].astype("int32").reshape(-1)
    y_train = np.load(folder+"train_outcome_" +name+".npz")["out_sex"].astype("int32").reshape(-1)
    
    best_model_file = folder + "model_sex_stratified.h5"
    output_file = folder + "prob_sex.npz"
    
    # testen
    best_model = keras.models.load_model(best_model_file,
                                         custom_objects={'ResBlock': ResBlock})
    
    vals = best_model.evaluate(X_test, y_test, verbose=0)
    print(dict(zip(best_model.metrics_names, [float(v) for v in vals])))
    
    # kansscores voor threshold-tuning
    probs_val  = best_model.predict(X_val,  verbose=0)[:, 0]
    probs_test = best_model.predict(X_test, verbose=0)[:, 0]
        
    prec, rec, thr = precision_recall_curve(y_val, probs_val)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    t_best = float(thr[np.nanargmax(f1)])
    
    y_hat = (probs_test >= t_best).astype("int32")
    print("ROC-AUC:", roc_auc_score(y_test, probs_test))
    print(classification_report(y_test, y_hat, digits=4))
    print(confusion_matrix(y_test, y_hat))
    
    np.savez(output_file, probs_val=probs_val, probs_test=probs_test, 
             thr=t_best, y_hat=y_hat)

