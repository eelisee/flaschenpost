# Tensorflow keras neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval
from sklearn.metrics import mean_squared_error, mean_absolute_error


target = "service_time_in_minutes"
features = ["article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", "num_previous_orders_customer", "customer_speed", "service_time_start"]

# Get Data
df_train, df_test = run_preprocessing()
X = df_train[features].astype(float)
y = df_train[target].astype(float)
X_test = df_test[features].astype(float)
y_test = df_test[target].astype(float)

# Define Model
model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Train model
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=32)

# Evaluate model
y_pred = model.predict(X_test)
print("Neural Network fitted. Evaluation on test-set:")
print(f"MSE = {mean_squared_error(y_test, y_pred)}")
print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
print(f"R2  = {1 - mean_squared_error(y_test, y_pred) / np.var(y_test)}")
print(f"Confidence Interval: {confidence_interval(y_pred)}")

# Save model to disk
import joblib
joblib.dump(model, './model/neuralnetwork.pkl')
print("Model saved to disk.")

