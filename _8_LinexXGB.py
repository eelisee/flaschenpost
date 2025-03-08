import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from _1_Preprocessing import run_preprocessing

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

def linex_loss_objective(y_pred, dtrain):
    """
    Custom objective function using Linex Loss.
    Wir definieren den Fehler als: error = y_pred - y
    Und verwenden den Linex-Loss:
        L(error) = exp(-a*error) + a*error - 1,
    sodass Unter‑Schätzungen (y_pred < y) stärker bestraft werden, wenn a > 0.
    Ableitungen:
        grad = a * (1 - exp(-a*error))
        hess = a^2 * exp(-a*error)
    """
    a = 1.0  # Parameter zur Steuerung der Asymmetrie (anpassbar)
    y = dtrain.get_label()
    error = y_pred - y
    grad = a * (1 - np.exp(-a * error))
    hess = a**2 * np.exp(-a * error)
    return grad, hess

def linex_xgb_regression():
    ########################################################################################################################
    # Standard XGBoost Regression mit benutzerdefiniertem Linex Loss
    print("Fitting XGBoost Regression with custom Linex Loss...")
    model = xgb.XGBRegressor(objective=linex_loss_objective, n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation auf dem Test-Set
    print("XGBoost Linex Regression fitted. Evaluation on test-set:")
    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
    print(f"R2  = {r2_score(y_test, y_pred)}")
    print("Train-Set Evaluation:")
    y_train_pred = model.predict(X_train_scaled)
    print(f"MSE = {mean_squared_error(y_train, y_train_pred)}, MAE = {mean_absolute_error(y_train, y_train_pred)}, R2 = {r2_score(y_train, y_train_pred)}")
    return model

def fine_tune_linex_xgb(X, y, param_grid):
    ########################################################################################################################
    # Fine-Tuning des XGBoost-Modells mit benutzerdefiniertem Linex Loss via GridSearchCV
    # Beispiel: param_grid = {"n_estimators": [50, 100, 150], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
    model = xgb.XGBRegressor(objective=linex_loss_objective, random_state=42)
    grid = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print("Best Hyperparameters:", grid.best_params_)
    print("Best Score (neg MSE):", grid.best_score_)
    return best_model, grid.best_params_, grid.best_score_

# Fine-tuning parameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}

# Fine-tuning auf den Trainingsdaten
best_model, best_params, best_score = fine_tune_linex_xgb(X_train_scaled, y_train, param_grid)
print("-----------------------------------------------")
print("Evaluating best model on Test-Set:")
y_pred_best = best_model.predict(X_test_scaled)
print(f"MSE = {mean_squared_error(y_test, y_pred_best)}")
print(f"MAE = {mean_absolute_error(y_test, y_pred_best)}")
print(f"R2  = {r2_score(y_test, y_pred_best)}")
print("-----------------------------------------------")
print("Fitting standard Linex XGBoost Regression...")
linex_xgb_regression()