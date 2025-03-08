import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pymc as pm
import arviz as az
from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval

target = "service_time_in_minutes"
features_to_keep = [
    "article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", 
    "num_previous_orders_customer", "customer_speed"
]

# Get Data
df_train, df_test = run_preprocessing()

# Add all columns that start with 'crate_count_' and 'article_id_'
features_to_keep += [col for col in df_train.columns if col.startswith("crate_count_")]
features_to_keep += [col for col in df_train.columns if col.startswith("article_id_")]

# Create a new dataframe with only the specified features
df_train_filtered = df_train[features_to_keep]
df_test_filtered = df_test[features_to_keep]

# Update X_train and X_test to use the filtered dataframes (as float arrays)
X_train = df_train_filtered.astype(float).values
y_train = df_train[target].astype(float).values
X_test = df_test_filtered.astype(float).values
y_test = df_test[target].astype(float).values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def bart_regression():
    ########################################################################################################################
    # BART Regression using PyMC's BART with fixed parameter m=50
    print("Fitting BART Regression...")
    with pm.Model() as model:
        # Use MutableData for flexible Datenübergabe
        X_shared = pm.Data("X", X_train_scaled)
        # BART: m=50 trees im Ensemble
        mu = pm.BART("mu", X_shared, y_train, m=50)
        sigma = pm.HalfNormal("sigma", sigma=1)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)
        trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True, progressbar=True)
    
    with model:
        pm.set_data({"X": X_test_scaled})
        pp = pm.sample_posterior_predictive(trace, var_names=["y_obs"], progressbar=True)
    y_pred = pp["y_obs"].mean(axis=0)
    
    # Evaluation on Test-Set
    print("BART Regression fitted. Evaluation on test-set:")
    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
    print(f"R2  = {r2_score(y_test, y_pred)}")
    print(f"Confidence Interval: {confidence_interval(y_pred)}")
    
    # Train-Set Evaluation (optional)
    with model:
        pm.set_data({"X": X_train_scaled})
        pp_train = pm.sample_posterior_predictive(trace, var_names=["y_obs"], progressbar=False)
    y_train_pred = pp_train["y_obs"].mean(axis=0)
    print("Train-Set Evaluation:")
    print(f"MSE = {mean_squared_error(y_train, y_train_pred)}, " +
          f"MAE = {mean_absolute_error(y_train, y_train_pred)}, " +
          f"R2 = {r2_score(y_train, y_train_pred)}, " +
          f"Confidence Interval: {confidence_interval(y_train_pred)}")
    print("-----------------------------------------------")

def fine_tune_bart(X, y, param_grid, test_size=0.2):
    ########################################################################################################################
    # Fine-Tuning des BART-Modells via manueller Grid Search (Hold-out-Validierung)
    # param_grid: z.B. {"m": [30, 50, 70], "n_iter": [500, 1000], "n_tune": [500, 1000]}
    best_score = float("inf")
    best_params = {}
    
    # Aufteilen in Trainings- und Validierungs-Sets
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    for m in param_grid.get("m", [50]):
        for n_iter in param_grid.get("n_iter", [500]):
            for n_tune in param_grid.get("n_tune", [500]):
                print(f"Evaluating BART with m = {m}, n_iter = {n_iter}, n_tune = {n_tune}...")
                with pm.Model() as model:
                    X_shared = pm.Data("X", X_train_sub)
                    mu = pm.BART("mu", X_shared, y_train_sub, m=m)
                    sigma = pm.HalfNormal("sigma", sigma=1)
                    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_sub)
                    trace = pm.sample(n_iter, tune=n_tune, target_accept=0.95, return_inferencedata=True, progressbar=False)
                with model:
                    pm.set_data({"X": X_val})
                    pp_val = pm.sample_posterior_predictive(trace, var_names=["y_obs"], progressbar=False)
                y_val_pred = pp_val["y_obs"].mean(axis=0)
                mse_val = mean_squared_error(y_val, y_val_pred)
                conf_int = confidence_interval(y_val_pred)
                print(f"m = {m}, n_iter = {n_iter}, n_tune = {n_tune}: Validation MSE = {mse_val}, Confidence Interval = {conf_int}")
                if mse_val < best_score:
                    best_score = mse_val
                    best_params = {"m": m, "n_iter": n_iter, "n_tune": n_tune}
                    best_conf_int = conf_int
    
    print("Best Hyperparameters:", best_params)
    print("Best (Validation) MSE:", best_score)
    print("Best Confidence Interval:", best_conf_int)
    return best_params, best_score

# Example parameter grid for fine-tuning
param_grid = {
    "m": [30, 50, 70],
    "n_iter": [500, 1000],
    "n_tune": [500, 1000]
}

# Fine-tuning on the training data
best_params, best_score = fine_tune_bart(X_train_scaled, y_train, param_grid, test_size=0.2)
print("Best Hyperparameters:", best_params)
print("Best (Validation) MSE:", best_score)
print("-----------------------------------------------")

# Final model training with best hyperparameters
print("Fitting final BART model with best hyperparameters...")
with pm.Model() as final_model:
    X_shared = pm.Data("X", X_train_scaled)
    mu = pm.BART("mu", X_shared, y_train, m=best_params["m"])
    sigma = pm.HalfNormal("sigma", sigma=1)
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)
    final_trace = pm.sample(1000, tune=1000, target_accept=0.95, return_inferencedata=True, progressbar=True)
with final_model:
    pm.set_data({"X": X_test_scaled})
    final_pp = pm.sample_posterior_predictive(final_trace, var_names=["y_obs"], progressbar=True)
y_final_pred = final_pp["y_obs"].mean(axis=0)
print("Final Model Evaluation on Test-Set:")
print(f"MSE = {mean_squared_error(y_test, y_final_pred)}")
print(f"MAE = {mean_absolute_error(y_test, y_final_pred)}")
print(f"R2  = {r2_score(y_test, y_final_pred)}")
print(f"Confidence Interval: {confidence_interval(y_final_pred)}")
print("-----------------------------------------------")

# Zum Abschluss die Standard-BART-Regression ausführen
bart_regression()
print("-----------------------------------------------")