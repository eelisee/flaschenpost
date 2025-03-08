import numpy as np

def confidence_interval(y_pred):
    n = len(y_pred)
    mu = np.mean(y_pred)
    sigma = np.std(y_pred, ddof=1)  # unbiased estimator (divides by n-1)
    z = sigma / np.sqrt(n)

    # For 95% CI, use Z-value 1.96 directly
    z_value = 1.96

    lower_bound = (z_value * z) - mu
    upper_bound = (z_value * z) + mu

    return lower_bound, upper_bound