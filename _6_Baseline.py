import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from _1_Preprocessing import run_preprocessing

target = "service_time_in_minutes"
#features = ["article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", "num_previous_orders_customer", "customer_speed", "service_time_start"]

df_train, df_test = run_preprocessing()

def baseline_evaluation():

    # Drop duplicate web order ids, keeping the first occurrence
    train_df_unique = df_train.drop_duplicates(subset='web_order_id')
    test_df_unique = df_test.drop_duplicates(subset='web_order_id')

    # Calculate the average service time in minutes over all unique web order ids
    average_service_time_train = train_df_unique[target].mean()
    average_service_time_test = test_df_unique[target].mean()

    # Calculate the average service time in minutes over all web order ids
    average_service_time_train = train_df_unique[target].mean()
    average_service_time_test = test_df_unique[target].mean()

    # Calculate MSE and MAE
    mse_train = mean_squared_error(train_df_unique[target], [average_service_time_train] * len(train_df_unique))
    mae_train = mean_absolute_error(train_df_unique[target], [average_service_time_train] * len(train_df_unique))
    mse_test = mean_squared_error(test_df_unique[target], [average_service_time_test] * len(test_df_unique))
    mae_test = mean_absolute_error(test_df_unique[target], [average_service_time_test] * len(test_df_unique))

    print("Baseline Evaluation:")
    print(f"Train-Set: MSE = {mse_train}, MAE = {mae_train}")
    print(f"Test-Set: MSE = {mse_test}, MAE = {mae_test}")

baseline_evaluation()
print("-----------------------------------------------")


# Baseline Evaluation:
# Train-Set: MSE = 36.403843976114636, MAE = 4.307491803247561
# Test-Set: MSE = 39.45388337665098, MAE = 4.501186744919033