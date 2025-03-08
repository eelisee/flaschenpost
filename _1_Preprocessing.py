import pandas as pd
import os

# BITTE DEN PFAD ANPASSEN (z.B. /data/)
root_path_data = 'data/'


def run_preprocessing(save=False):
    df_orders, df_driver_order_mapping, df_service_times, df_order_articles = load_data()
    df = merge_tables(df_orders, df_driver_order_mapping, df_service_times)
    # df = pd.merge(df, df_order_articles[['web_order_id', 'article_id']], on='web_order_id', how='left')
    print("size before remove outliers: ", df.shape)
    df = remove_outliers(df)
    print("size after remove outliers: ", df.shape)
    df = add_article_total_weight(df, df_order_articles)
    print("size after add article total weight: ", df.shape)
    # df = one_hot_encoding(df)
    df = add_crate_counts(df, df_order_articles)
    print("size after add crate count: ", df.shape)
    # df = one_hot_encoding(df, ["warehouse_id", "driver_id"])
    df = handle_missing_values(df)
    print("size after handle missing values: ", df.shape)
    df = service_time_start_ordinal_encoding(df)
    print("size after service time start ordinal encoding: ", df.shape)
    df_train, df_test = train_test_split(df)
    print("size after train test split: ", df_train.shape, df_test.shape)
    df_train, df_test = add_num_previous_orders__per_customer(df_train, df_test)
    df_train, df_test = add_customer_speed_ordinal(df_train, df_test)
    if save:
        save_data_to_parquet(df_train, df_test)
    return df_train, df_test

def save_data_to_parquet(df_train, df_test):
    df_train.to_parquet(os.path.join(root_path_data, "df_train.parquet"), engine='fastparquet')
    df_test.to_parquet(os.path.join(root_path_data, "df_test.parquet"), engine='fastparquet')

def load_data_from_parquet():
    df_train = pd.read_parquet(os.path.join(root_path_data, "df_train.parquet"), engine='fastparquet')
    df_test = pd.read_parquet(os.path.join(root_path_data, "df_test.parquet"), engine='fastparquet')
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
    article_total_weight = df_order_articles[["article_weight_in_g", "web_order_id", "box_id"]].groupby("web_order_id").sum()
    df = pd.merge(df, article_total_weight, on="web_order_id", how='left')
    return df


def one_hot_encoding(df):
    article_ids_to_encode = [15043, 20619, 18544, 21243]

    # One hot encode article_id
    article_id_dummies = df.groupby('web_order_id')['article_id'].apply(lambda x: pd.Series(
        {f'article_id_{article_id}': 1 for article_id in article_ids_to_encode if article_id in x.values}))
    article_id_dummies = article_id_dummies.unstack().fillna(0).reset_index()
    df = pd.merge(df, article_id_dummies, on='web_order_id', how='left')
    missing_article_columns = {f'article_id_{article_id}': 0 for article_id in article_ids_to_encode if
                               f'article_id_{article_id}' not in df.columns}
    df = df.assign(**missing_article_columns)
    # Fill missing article columns with 0 (only article_id_* cols)
    df[df.columns[df.columns.str.contains("article_id")]] = df[df.columns[df.columns.str.contains("article_id")]].fillna(0)

def add_crate_counts(df,df_order_articles):
    crate_counts_to_encode = [60, 45, 42, 41, 43, 44, 46, 47, 39, 37, 35, 38, 40, 50, 48, 36, 33, 34, 31, 52, 32, 49,
                              28, 30, 29, 27]

    # Merge box ids to orders
    df_crate_counts = pd.merge(df, df_order_articles[['web_order_id', 'box_id']], on='web_order_id', how='left', suffixes=('', '_y'))
    df_crate_counts.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)
    df_crate_counts = df_crate_counts[['web_order_id', 'box_id']]

    # Group by web_order_id
    # group = df_crate_counts.groupby('web_order_id')
    # df_crate_counts["crate_count"] = group.agg({'box_id': 'nunique'}).reset_index()['box_id']
    # df_crate_counts["crate_count"] += group["box_id"].transform(lambda x: x.isnull().sum())
    # df_crate_counts = df_crate_counts[['web_order_id', 'crate_count']]
    # # Merge duplicates
    # df_crate_counts = df_crate_counts.drop_duplicates()

    df_tmp = df_crate_counts.copy()
    df_tmp['box_id'] = df_tmp['box_id'].fillna(0)
    nan_boxes_df = df_tmp[df_tmp['box_id'] == 0]
    drink_count = nan_boxes_df.groupby('web_order_id').count()
    df_tmp = df_crate_counts.dropna(axis=0, subset=['box_id'])
    drink_count['food_boxes'] = df_tmp.groupby('web_order_id')['box_id'].nunique()
    drink_count = drink_count.fillna(0)
    df['crate_count'] = drink_count['food_boxes'] + drink_count['box_id']


    # Count NaNs in 'box id' per 'order id' and store in 'crate count'
    # df['box_count'] = df.groupby('web_order_id')['box_id'].transform(lambda x: x.isna().sum())
    # df['crate_count'] = df.groupby('web_order_id')['box_id'].transform(lambda x: x.nunique()) + df['box_count']

    # One hot encode crate_count
    # df['crate_count'] = df['crate_count'].astype(str)
    # df = pd.get_dummies(df, columns=['crate_count'], prefix='crate_count', prefix_sep='_')
    # # missing_crate_columns = {f'crate_count_{crate_count}': 0 for crate_count in crate_counts_to_encode if
    # #                          f'crate_count_{crate_count}' not in df.columns}
    # df = df.assign(**missing_crate_columns)
    # merge back to df
    # df = pd.merge(df, drink_count[['web_order_id', 'crate_count']], on='web_order_id', how='left')
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

def subsample_for_plotting(df, n=100_000):
    return df.sample(n, random_state=0)


if __name__ == '__main__':
    run_preprocessing()
    print("Preprocessing finished.")