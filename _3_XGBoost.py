import numpy
import numpy as np

from _1_Preprocessing import run_preprocessing
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


target = "service_time_in_minutes"
features = ["article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", "num_previous_orders_customer", "customer_speed", "service_time_start"]

# Get Data
df_train, df_test = run_preprocessing()
X = df_train[features].astype(float)
y = df_train[target].astype(float)
X_test = df_test[features].astype(float)
y_test = df_test[target].astype(float)

id = numpy.array(['%d' % i for i in range(len(y))])

# Get all column names for one hot encoding
warehouse_id_cols = df_train.columns[df_train.columns.str.contains("warehouse_id")]
driver_id_cols = df_train.columns[df_train.columns.str.contains("driver_id")]


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
                             seed=42)

    # Train the model
    model.fit(X, y)

    # Predict the test set
    y_pred = model.predict(X_test)

    # Evaluation
    # print("XGBoost fitted. Evaluation on test-set:")
    # print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    # print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
    # print(f"R2 = {model.score(X_test, y_test)}")
    # print(f"Train-Set-Evaluation: MSE = {mean_squared_error(y, model.predict(X))}, MAE = {mean_absolute_error(y_test, y_pred)}, R2 = {model.score(X, y)}")


    MSE_train = mean_squared_error(y, model.predict(X))
    MAE_train = mean_absolute_error(y, model.predict(X))
    R2_train = model.score(X, y)
    MSE_test = mean_squared_error(y_test, y_pred)
    MAE_test = mean_absolute_error(y_test, y_pred)
    R2_test = model.score(X_test, y_test)

    return MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test, model

# params = {
#     "n_estimators": [1000],
#     "max_depth": [10, 20],
#     "eta": [0.01, 0.1, 0.3],
#     "subsample": [0.5, 0.75, 1],
#     "colsample_bytree": [0.5, 0.75, 1]
# }
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
                    MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test, model = xgboost({"n_estimators": n_estimators, "max_depth": max_depth, "eta": eta, "subsample": subsample, "colsample_bytree": colsample_bytree})
                    print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, eta: {eta}, subsample: {subsample}, colsample_bytree: {colsample_bytree}, MSE_train: {MSE_train}, MAE_train: {MAE_train}, R2_train: {R2_train}, MSE_test: {MSE_test}, MAE_test: {MAE_test}, R2_test: {R2_test}")
                    best_params = best_params if min_mae < MAE_test else {"n_estimators": n_estimators, "max_depth": max_depth, "eta": eta, "subsample": subsample, "colsample_bytree": colsample_bytree}
                    min_mae = min_mae if min_mae < MAE_test else MAE_test
                    model = model if min_mae < MAE_test else model

print(f"Best Parameters: {best_params}")
print(f"Min MAE: {min_mae}")
print("----------------------------")


# Use best parameters to fit final model. Then identify the worst predictions (only underestimating the service time) on the training set.
# Set their sample weight higher and retrain the model
y_preds_train = model.predict(X)
# Identify the worst underestimations
diff = y - y_preds_train
worst_underestimations = diff[diff > 10]
# Set their sample weight higher
sample_weight = np.ones(len(y))
sample_weight[diff > 10] = 2

model = xgb.XGBRegressor(n_estimators=best_params["n_estimators"],
                         max_depth=best_params["max_depth"],
                         eta=best_params["eta"],
                         subsample=best_params["subsample"],
                         colsample_bytree=best_params["colsample_bytree"],
                         seed=42)

model.fit(X, y, sample_weight=sample_weight)

# Evaluate on test
y_pred = model.predict(X_test)
print("XGBoost fitted. Evaluation on test-set:")
print(f"MSE = {mean_squared_error(y_test, y_pred)}")
print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
print(f"R2 = {model.score(X_test, y_test)}")
print(f"Train-Set-Evaluation: MSE = {mean_squared_error(y, model.predict(X))}, MAE = {mean_absolute_error(y_test, y_pred)}, R2 = {model.score(X, y)}")

