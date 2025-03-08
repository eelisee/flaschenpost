import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from _1_Preprocessing import run_preprocessing

# Get Data
df_train, df_test = run_preprocessing()

# Define the list of features to keep
features_to_keep = [
    "article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", 
    "num_previous_orders_customer", "customer_speed"
]

# Add all columns that start with 'crate_count_' and 'article_id_'
features_to_keep += [col for col in df_train.columns if col.startswith('crate_count_')]
features_to_keep += [col for col in df_train.columns if col.startswith('article_id_')]

# Create a new dataframe with only the specified features
df_train_filtered = df_train[features_to_keep]
df_test_filtered = df_test[features_to_keep]

# Update X_train and X_test to use the filtered dataframes
X_train = df_train_filtered
X_test = df_test_filtered

# Apply StandardScaler only on continuous variables
continuous_features = [
    "article_weight_in_g", "floor", "num_previous_orders_customer", "customer_speed"
]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

# Assuming 'service_time_in_minutes' is the target variable and the rest are features
y_train = df_train['service_time_in_minutes']
y_test = df_test['service_time_in_minutes']

def bayesian_ridge_regression():
    ########################################################################################################################
    # Bayesian Ridge Regression
    print("Fitting Bayesian Ridge Regression...")
    model = BayesianRidge()
    model.fit(X_train_scaled, y_train)

    # Predict the values for the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    print("Bayesian Ridge Regression fitted. Evaluation on test-set:")
    print(f"MSE = {mean_squared_error(y_test, y_pred)}")
    print(f"MAE = {mean_absolute_error(y_test, y_pred)}")
    print(f"R2 = {model.score(X_test_scaled, y_test)}")
    print(f"Train-Set-Evaluation: MSE = {mean_squared_error(y_train, model.predict(X_train_scaled))}, MAE = {mean_absolute_error(y_train, model.predict(X_train_scaled))}, R2 = {model.score(X_train_scaled, y_train)}")

bayesian_ridge_regression()
print("-----------------------------------------------")

def bayesian_ridge_regression_with_grid_search():
    ########################################################################################################################
    # Bayesian Ridge Regression with Grid Search
    print("Fitting Bayesian Ridge Regression with Grid Search...")
    param_grid = {
        'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
        'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3]
    }

    model = BayesianRidge()
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1 
    )

    grid_search.fit(X_train_scaled, y_train)

    # Output the best hyperparameters and the corresponding score
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best CV MSE:", -grid_search.best_score_)

    # Iterate over the results and print the MSE and MAE for each configuration
    for i, params in enumerate(grid_search.cv_results_['params']):
        print(f"Configuration {i+1}: {params}")
        mse = -grid_search.cv_results_['mean_test_score'][i]
        print(f"Mean Squared Error: {mse}")
        # Fit the model with the current parameters
        model.set_params(**params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae}")

bayesian_ridge_regression_with_grid_search()
print("-----------------------------------------------")
