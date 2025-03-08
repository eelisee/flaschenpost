import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval
from scipy import stats

def confidence_interval(y_true, y_pred, confidence=0.95):
    residuals = y_true - y_pred
    n = len(residuals)
    mean_error = np.mean(residuals)
    se = stats.sem(residuals)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    lower_bound = mean_error - h
    upper_bound = mean_error + h
    return lower_bound, upper_bound


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

# Create filtered DataFrames
df_train_filtered = df_train[features_to_keep]
df_test_filtered = df_test[features_to_keep]

# Update X_train and X_test to use the filtered DataFrames (as float arrays)
X_train = df_train_filtered.astype(float).values
y_train = df_train[target].astype(float).values
X_test = df_test_filtered.astype(float).values
y_test = df_test[target].astype(float).values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def linex_loss_lgb(y_true, y_pred):
    a = 1.0  # Customize as needed
    error = y_pred - y_true
    grad = a * (1 - np.exp(-a * error))
    hess = a**2 * np.exp(-a * error)
    return grad, hess

def linex_lgb_regression():
    print("Fitting LightGBM Regression with custom Linex Loss...")
    model = lgb.LGBMRegressor(objective=linex_loss_lgb, n_estimators=100, learning_rate=0.1, random_state=0)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    print("Test set results:")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    print("Confidence Interval:", confidence_interval(y_test, y_pred))

    return model

# Fine-tuning function
def fine_tune_linex_lgb(X, y, param_grid):
    model = lgb.LGBMRegressor(objective=linex_loss_lgb)
    grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    print("Best Params:", grid.best_params_)
    print("Best Score (neg MSE):", grid.best_score_)
    return best_model, grid.best_params_, grid.best_score_

# Fine-tuning parameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}

# Call fine-tuning
#best_model, best_params, best_score = fine_tune_linex_lgb(X_train_scaled, y_train, param_grid)

# Final model with best parameters
final_model = lgb.LGBMRegressor(objective=linex_loss_lgb, learning_rate=0.2, max_depth=3, n_estimators=50, random_state=0)
final_model.fit(X_train_scaled, y_train)

y_pred_final = final_model.predict(X_test_scaled)
print("Final model results:")
print("MSE:", mean_squared_error(y_test, y_pred_final))
print("MAE:", mean_absolute_error(y_test, y_pred_final))
print("R²:", r2_score(y_test, y_pred_final))
print("Confidence Interval:", confidence_interval(y_test, y_pred_final))

# Save model to disk
import joblib
joblib.dump(final_model, './model/light_gbm.pkl')
print("Model saved to disk.")


# LightGBM Linex Regression fitted. Evaluation on test-set:
# Final model results:
# MSE: 143.71673070052464
# MAE: 9.736442076663906
# R²: -1.937721030387189
