import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from GRANDE import GRANDE
from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval
from sklearn.naive_bayes import GaussianNB

target = "service_time_in_minutes"
features = ["article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", "num_previous_orders_customer", "customer_speed", "service_time_start"]

# Get Data
df_train, df_test = run_preprocessing()
X = df_train[features].astype(float)
y = df_train[target].astype(float)
X_test = df_test[features].astype(float)
y_test = df_test[target].astype(float)

# Get all column names for one hot encoding
warehouse_id_cols = df_train.columns[df_train.columns.str.contains("warehouse_id")]
driver_id_cols = df_train.columns[df_train.columns.str.contains("driver_id")]


def linear_regression():
    ########################################################################################################################
    # Linear Regression
    X = df_train[features].astype(float)
    y = df_train[target].astype(float)
    print("Fitting Linear Regression...")
    model = LinearRegression()
    model.fit(X, y)


    y_pred = model.predict(X_test)

    # Evaluation
    print("Linear Regression fitted. Evaluation on test-set:")
    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
    print(f"R2 = {model.score(X_test, y_test)}")
    print(f"Confidence Interval: {confidence_interval(y_pred)}")
    print(f"Train-Set-Evaluation: MSE = {mean_squared_error(y, model.predict(X))}, MAE = {mean_absolute_error(y_test, y_pred)}, R2 = {model.score(X, y)}")

    # Save model to disk ./model/linear_regression.pkl
    import joblib
    joblib.dump(model, './model/linear_regression.pkl')


linear_regression()
print("-----------------------------------------------")

def ridge_regression(params):
    ########################################################################################################################
    # Ridge Regression
    print("Fitting Ridge Regression...")
    model = Ridge(alpha=params["alpha"],
                  solver=params["solver"],
                  tol=params["tol"],
                  max_iter=params["max_iter"],
                  random_state=params["random_state"])

    model.fit(X, y)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    # print("Lasso Regression fitted. Evaluation on test-set:")
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
    conf_int = confidence_interval(y_pred)

    return MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test, conf_int

param_jobs = {"alpha": [0.01, 0.1, 0.5, 1.0, 1.5, 2, 10],
          "solver": ['auto'],
          "tol": [0.001, 0.01, 0.1, 1],
          "max_iter": None,
          "random_state": 42}

# Loop through all combinations of hyperparameters
min_mae = float("inf")
min_mae_params = {}
params = param_jobs.copy()
for alpha in param_jobs["alpha"]:
    for solver in param_jobs["solver"]:
        for tol in param_jobs["tol"]:
            params["alpha"] = alpha
            params["solver"] = solver
            params["tol"] = tol
            MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test =  ridge_regression(params)
            print(f"alpha={alpha}, solver={solver}, tol={tol}, MSE_train={MSE_train}, MAE_train={MAE_train}, R2_train={R2_train}, MSE_test={MSE_test}, MAE_test={MAE_test}, R2_test={R2_test}")
            min_mae = min_mae if min_mae < MAE_test else MAE_test
            min_mae_params = min_mae_params if min_mae < MAE_test else params.copy()

# Best Hyperparameters
print(f"Best Hyperparameters: {min_mae_params}")
print(f"Best MAE: {min_mae}")
print("-----------------------------------------------")

def lasso_regression(params):
    ########################################################################################################################
    # Lasso Regression
    print("Fitting Lasso Regression...")
    model = Lasso(alpha=params["alpha"],
                  tol=params["tol"],
                  max_iter=params["max_iter"],
                  random_state=params["random_state"])
    model.fit(X, y)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    # print("Hinge Regression fitted. Evaluation on test-set:")
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
    conf_int = confidence_interval(y_pred)

    return MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test, conf_int

param_jobs = {"alpha": [0.01, 0.1, 0.5, 1.0, 1.5, 2, 10],
              "solver": ['auto'],
              "tol": [0.0001, 0.001, 0.01, 0.1, 1],
              "max_iter": 1000,
              "random_state": 42}

# Loop through all combinations of hyperparameters
params = param_jobs.copy()
for alpha in param_jobs["alpha"]:
    for tol in param_jobs["tol"]:
        params["alpha"] = alpha
        params["tol"] = tol
        MSE_train, MAE_train, R2_train, MSE_test, MAE_test, R2_test, conf_int = lasso_regression(params)
        
        print(
            f"alpha={alpha}, solver={solver}, tol={tol}, MSE_train={MSE_train}, MAE_train={MAE_train}, R2_train={R2_train}, MSE_test={MSE_test}, MAE_test={MAE_test}, R2_test={R2_test}, Confidence Interval={conf_int}")
        min_mae = min_mae if min_mae < MAE_test else MAE_test
        min_mae_params = min_mae_params if min_mae < MAE_test else params.copy()

# Best Hyperparameters
print(f"Best Hyperparameters: {min_mae_params}")
print(f"Best MAE: {min_mae}")
print("-----------------------------------------------")
