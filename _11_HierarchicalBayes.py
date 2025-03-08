import numpy as np
import pymc as pm
import arviz as az
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval

# Zielvariable und Featureliste
target = "service_time_in_minutes"
features_to_keep = [
    "article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor",
    "num_previous_orders_customer", "customer_speed"
]

# Get Data
df_train, df_test = run_preprocessing()

# Ergänze Features, die mit 'crate_count_' und 'article_id_' beginnen
features_to_keep += [col for col in df_train.columns if col.startswith("crate_count_")]
features_to_keep += [col for col in df_train.columns if col.startswith("article_id_")]

# Erstelle gefilterte DataFrames
df_train_filtered = df_train[features_to_keep]
df_test_filtered = df_test[features_to_keep]

# Konvertiere in float und extrahiere X und y
X_train = df_train_filtered.astype(float).values
y_train = df_train[target].astype(float).values
X_test = df_test_filtered.astype(float).values
y_test = df_test[target].astype(float).values

# Skalieren der Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Gruppierungsvariablen: Wir gehen davon aus, dass in df_train und df_test Spalten "customer_id" und "driver_id" existieren.
customer_ids_train = df_train['customer_id'].astype('category').cat.codes.values
driver_ids_train   = df_train['driver_id'].astype('category').cat.codes.values
customer_ids_test  = df_test['customer_id'].astype('category').cat.codes.values
driver_ids_test    = df_test['driver_id'].astype('category').cat.codes.values

n_customers = len(np.unique(customer_ids_train))
n_drivers   = len(np.unique(driver_ids_train))
n_features  = X_train_scaled.shape[1]

###########################################################################
# Hierarchisches Bayesian Modell (Regression) mit Kunden- und Fahrer-Effekten
###########################################################################
def hierarchical_regression():
    print("Fitting hierarchical Bayesian regression model...")
    with pm.Model() as model:
        # Globale Effekte: Intercept und Koeffizienten
        beta0 = pm.Normal("beta0", mu=0, sigma=10)
        beta  = pm.Normal("beta", mu=0, sigma=10, shape=n_features)
        
        # Gruppen-spezifische Effekte
        sigma_customer = pm.HalfNormal("sigma_customer", sigma=5)
        sigma_driver   = pm.HalfNormal("sigma_driver", sigma=5)
        customer_effect = pm.Normal("customer_effect", mu=0, sigma=sigma_customer, shape=n_customers)
        driver_effect   = pm.Normal("driver_effect", mu=0, sigma=sigma_driver, shape=n_drivers)
        
        # Erwartungswert der Service-Time:
        mu_val = beta0 + pm.math.dot(X_train_scaled, beta) + customer_effect[customer_ids_train] + driver_effect[driver_ids_train]
        
        # Beobachtungsmodell
        sigma = pm.HalfNormal("sigma", sigma=5)
        y_obs = pm.Normal("y_obs", mu=mu_val, sigma=sigma, observed=y_train)
        
        # Sampling aus dem Posterior
        trace = pm.sample(10, tune=20, target_accept=0.95, return_inferencedata=True, progressbar=True)
    
    # Vorhersagen auf Testdaten
    with model:
        mu_test = beta0 + pm.math.dot(X_test_scaled, beta) + customer_effect[customer_ids_test] + driver_effect[driver_ids_test]
        y_obs_test = pm.Normal("y_obs_test", mu=mu_test, sigma=sigma, observed=y_test)
        # Da wir ein normales Likelihood-Modell verwenden, nutzen wir den Mittelwert als Vorhersage
        y_pred = posterior_pred['y_obs_test'].mean(axis=0)
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y_obs"], progressbar=True)
    
    y_pred = posterior_pred[y_obs].mean(axis=0)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    conf_int = confidence_interval(y_pred)
    
    print("Hierarchisches Modell fitted. Evaluation on Test-Set:")
    print(f"MSE = {mse}")
    print(f"MAE = {mae}")
    print(f"R2  = {r2}")
    print(f"Confidence Interval: {conf_int}")

    # Save model to disk
    import joblib
    joblib.dump(model, './model/hierarchical_bayes.pkl')
    print("Model saved to disk.")
    
    return model, trace, y_pred

if __name__ == '__main__':
    hierarchical_regression()