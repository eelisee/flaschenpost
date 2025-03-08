import numpy
import numpy as np

from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


target = "service_time_in_minutes"
features = ["article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", "num_previous_orders_customer", "customer_speed", "service_time_start"]

# Get Data
df_train, df_test = run_preprocessing()
X = df_train[features].astype(float)
y = df_train[target].astype(float)
X_test = df_test[features].astype(float)
y_test = df_test[target].astype(float)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

id = numpy.array(['%d' % i for i in range(len(y))])

# Get all column names for one hot encoding
warehouse_id_cols = df_train.columns[df_train.columns.str.contains("warehouse_id")]
driver_id_cols = df_train.columns[df_train.columns.str.contains("driver_id")]

# Define the custom Linex loss function for XGBoost
def linex_obj(y_pred, dtrain):
    a = 1.0  # Adjust this parameter based on your preference
    y_true = dtrain.get_label()
    error = y_pred - y_true
    grad = a * (1 - np.exp(-a * error))
    hess = (a**2) * np.exp(-a * error)
    return grad, hess


def xgboost(params):
    ########################################################################################################################
    # XGBoost
    print("Fitting XGBoost...")
    # Define XGBoost regressor
    model = xgb.XGBRegressor(n_estimators=params["n_estimators"],
                             max_depth=params["max_depth"],
                             eta=params["eta"],
                             subsample=params["subsample"],
                             colsample_bytree=params["colsample_bytree"],
                             seed=42,
                             n_threads=8)

    # Train the model
    model.fit(X, y)

    # Predict the test set
    y_pred_valid = model.predict(X_valid)

    # Evaluation
    MSE_train = mean_squared_error(y_train, model.predict(X_train))
    MAE_train = mean_absolute_error(y_train, model.predict(X_train))
    R2_train = model.score(X_train, y_train)
    MSE_valid = mean_squared_error(y_valid, y_pred_valid)
    MAE_valid = mean_absolute_error(y_valid, y_pred_valid)
    R2_valid = model.score(X_valid, y_valid)
    conf_int = confidence_interval(y_pred_valid)

    return MSE_train, MAE_train, R2_train, MSE_valid, MAE_valid, R2_valid, conf_int, model

params = {
    "n_estimators": [1000],
    "max_depth": [10],
    "eta": [0.3],
    "subsample": [0.75],
    "colsample_bytree": [0.5]
}

min_mae = np.inf
best_params = {}
model = None
for n_estimators in params["n_estimators"]:
    for max_depth in params["max_depth"]:
        for eta in params["eta"]:
            for subsample in params["subsample"]:
                for colsample_bytree in params["colsample_bytree"]:
                    MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test, conf_int, model = xgboost({"n_estimators": n_estimators, "max_depth": max_depth, "eta": eta, "subsample": subsample, "colsample_bytree": colsample_bytree})
                    print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, eta: {eta}, subsample: {subsample}, colsample_bytree: {colsample_bytree}, MSE_train: {MSE_train}, MAE_train: {MAE_train}, R2_train: {R2_train}, MSE_test: {MSE_test}, MAE_test: {MAE_test}, R2_test: {R2_test}, conf_int: {conf_int}")
                    best_params = best_params if min_mae < MAE_test else {"n_estimators": n_estimators, "max_depth": max_depth, "eta": eta, "subsample": subsample, "colsample_bytree": colsample_bytree}
                    min_mae = min_mae if min_mae < MAE_test else MAE_test
                    model = model if min_mae < MAE_test else model

print(f"Best Parameters: {best_params}")
print(f"Min MAE (valid): {min_mae}")
print("----------------------------")


df_train, df_test = run_preprocessing()

model = xgb.XGBRegressor(n_estimators=best_params["n_estimators"],
                         max_depth=best_params["max_depth"],
                         eta=best_params["eta"],
                         subsample=best_params["subsample"],
                         colsample_bytree=best_params["colsample_bytree"],
                         objective=linex_obj,
                         seed=42)

model.fit(X, y,
          # sample_weight=sample_weight
          )

# Evaluate on test
y_pred = model.predict(X_test)
print("XGBoost fitted. Evaluation on test-set:")
print(f"MSE = {mean_squared_error(y_test, y_pred)}")
print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
print(f"R2 = {model.score(X_test, y_test)}")
print(f"Confidence Interval: {confidence_interval(y_pred)}")
print(f"Train-Set-Evaluation: MSE = {mean_squared_error(y, model.predict(X))}, MAE = {mean_absolute_error(y_test, y_pred)}, R2 = {model.score(X, y)}, Confidence Interval: {confidence_interval(model.predict(X))}")

print("Dataset-size (train): ", len(df_train))
print("Dataset-size (test): ", len(df_test))
print("Number of features: ", len(features))
print("Feature-selection: ", features)

# Save model to disk
import joblib
joblib.dump(model, './model/linex_regression.pkl')
print("Model saved to disk.")
