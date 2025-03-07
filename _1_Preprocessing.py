import pandas as pd
import os

# BITTE DEN PFAD ANPASSEN (z.B. /data/)
root_path_data = 'data/'


def run_preprocessing():
    df_orders, df_driver_order_mapping, df_service_times, df_order_articles = load_data()
    df = merge_tables(df_orders, df_driver_order_mapping, df_service_times)
    df = pd.merge(df, df_order_articles[['web_order_id', 'article_id']], on='web_order_id', how='left')
    df = remove_outliers(df)
    df = add_article_total_weight(df, df_order_articles)
    df = one_hot_encoding(df)
    # df = one_hot_encoding(df, ["warehouse_id", "driver_id"])
    df = handle_missing_values(df)
    df = service_time_start_ordinal_encoding(df)
    df_train, df_test = train_test_split(df)
    df_train, df_test = add_num_previous_orders__per_customer(df_train, df_test)
    df_train, df_test = add_customer_speed_ordinal(df_train, df_test)
    return df_train, df_test


def load_data():
    df_orders = pd.read_parquet(os.path.join(root_path_data, "masked_orders.parquet"))
    df_driver_order_mapping = pd.read_parquet(os.path.join(root_path_data, 'masked_driver_order_mapping.parquet'))
    df_service_times = pd.read_parquet(os.path.join(root_path_data, 'masked_service_times.parquet'))
    df_order_articles = pd.read_parquet(os.path.join(root_path_data, 'masked_order_articles.parquet'))
    return df_orders, df_driver_order_mapping, df_service_times, df_order_articles


def merge_tables(df_orders, df_driver_order_mapping, df_service_times):
    df = pd.merge(df_orders, df_service_times, on="web_order_id", how='left', suffixes=('', '_y'))
    df = pd.merge(df, df_driver_order_mapping, on="web_order_id", how='left', suffixes=('', '_y'))
    df.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)
    return df


def add_article_total_weight(df, df_order_articles):
    article_total_weight = df_order_articles[["article_weight_in_g", "web_order_id"]].groupby("web_order_id").sum()
    df = pd.merge(df, article_total_weight, on="web_order_id", how='left')
    return df


def one_hot_encoding(df):
    article_ids_to_encode = [15043, 20619, 18544, 21243]
    crate_counts_to_encode = [60, 45, 42, 41, 43, 44, 46, 47, 39, 37, 35, 38, 40, 50, 48, 36, 33, 34, 31, 52, 32, 49,
                              28, 30, 29, 27]
    # One hot encode article_id
    article_id_dummies = df.groupby('web_order_id')['article_id'].apply(lambda x: pd.Series(
        {f'article_id_{article_id}': 1 for article_id in article_ids_to_encode if article_id in x.values}))
    article_id_dummies = article_id_dummies.unstack().fillna(0).reset_index()
    df = pd.merge(df, article_id_dummies, on='web_order_id', how='left')
    missing_article_columns = {f'article_id_{article_id}': 0 for article_id in article_ids_to_encode if
                               f'article_id_{article_id}' not in df.columns}
    df = df.assign(**missing_article_columns)

    # Calculate crate_count by summing the number of unique box_ids + the number of rows that have box_id NaN per web_order_id
    if 'box_id' in df.columns:
        crate_count = df.groupby("web_order_id").apply(lambda x: x["box_id"].nunique() + x["box_id"].isna().sum())
        crate_count = crate_count.reset_index(name="crate_count")
        df = pd.merge(df, crate_count, on="web_order_id", how="left")
    else:
        df['crate_count'] = 0

    # One hot encode crate_count
    df['crate_count'] = df['crate_count'].astype(str)
    df = pd.get_dummies(df, columns=['crate_count'], prefix='crate_count', prefix_sep='_')
    missing_crate_columns = {f'crate_count_{crate_count}': 0 for crate_count in crate_counts_to_encode if
                             f'crate_count_{crate_count}' not in df.columns}
    df = df.assign(**missing_crate_columns)

    return df


def handle_missing_values(df):
    df = df.dropna()
    return df


def service_time_start_ordinal_encoding(df):
    df["service_time_start"] = pd.to_datetime(df["service_time_start"])
    df["service_time_start"] = df["service_time_start"].dt.hour
    return df


def train_test_split(df):
    df_train = df.sample(frac=0.8, random_state=0)
    df_test = df.drop(df_train.index)
    return df_train, df_test


def add_num_previous_orders__per_customer(df_train, df_test):
    num_previous_orders_customer = df_train["customer_id"].value_counts().to_dict()
    df_train["num_previous_orders_customer"] = df_train["customer_id"].map(num_previous_orders_customer).fillna(0)
    df_test["num_previous_orders_customer"] = df_test["customer_id"].map(num_previous_orders_customer).fillna(0)
    return df_train, df_test


def add_customer_speed_ordinal(df_train, df_test):
    # Count the number of orders per customer
    customer_order_counts = df_train["customer_id"].value_counts()
    # Filter customers with more than 5 orders
    customers_with_more_than_5_orders = customer_order_counts[customer_order_counts > 5].index
    # Calculate the average service time for these customers
    customer_avg_service_time = df_train[df_train["customer_id"].isin(customers_with_more_than_5_orders)][
        ["customer_id", "service_time_in_minutes"]].groupby("customer_id").mean()

    customer_avg_service_time["customer_speed"] = pd.cut(customer_avg_service_time["service_time_in_minutes"], bins=9,
                                                         labels=False)
    df_train["customer_speed"] = df_train["customer_id"].map(customer_avg_service_time["customer_speed"]).fillna(4)
    df_test["customer_speed"] = df_test["customer_id"].map(customer_avg_service_time["customer_speed"]).fillna(4)
    return df_train, df_test

def remove_outliers(df):
    # Remove outliers
    df = df[df["service_time_in_minutes"] < 60]
    df = df[df["floor"] < 50]
    return df
