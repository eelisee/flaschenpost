import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from GRANDE import GRANDE
from _1_Preprocessing import run_preprocessing


# Get Data
df_train, df_test = run_preprocessing()

# Get all column names for one hot encoding
warehouse_id_cols = df_train.columns[df_train.columns.str.contains("warehouse_id")]
driver_id_cols = df_train.columns[df_train.columns.str.contains("driver_id")]

target = "service_time_in_minutes"
features = ["article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", "num_previous_orders_customer", "customer_speed"]
# features.extend(warehouse_id_cols)
# features.extend(driver_id_cols)

X = df_train[features].astype(float)
y = df_train[target].astype(float)

print("Fitting Linear Regression...")
model = LinearRegression()
model.fit(X, y)

# Prediction
X_test = df_test[features].astype(float)
y_test = df_test[target].astype(float)

y_pred = model.predict(X_test)

# Evaluation
print("Linear Regression fitted. Evaluation on test-set:")
print(f"MSE = {mean_squared_error(y_test, y_pred)}")
print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
print(f"R2 = {model.score(X_test, y_test)}")