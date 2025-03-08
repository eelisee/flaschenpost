import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGP, AbstractDeepGPLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from _1_Preprocessing import run_preprocessing
from _12_evaluation import confidence_interval

# Zielvariable und Features definieren
target = "service_time_in_minutes"

# Get Data
df_train, df_test = run_preprocessing()

# After train-test split:
max_total_rows = 1000  # safe small size to begin

# Sample order_ids randomly until reaching the desired max_total_rows
unique_order_ids = df_train['web_order_id'].unique()
np.random.shuffle(unique_order_ids)

sampled_order_ids = []
total_rows = 0

for oid in unique_order_ids:
    oid_rows = df_train[df_train['web_order_id'] == oid].shape[0]
    if total_rows + oid_rows > max_total_rows:
        break
    sampled_order_ids.append(oid)
    total_rows += oid_rows

# Subset df_train based on sampled_order_ids
df_train_subset = df_train[df_train['web_order_id'].isin(sampled_order_ids)].copy()

# Now proceed with feature selection AFTER subsetting:
features_to_keep = [
    "article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "floor", 
    "num_previous_orders_customer", "customer_speed"
]
features_to_keep += [col for col in df_train_subset.columns if col.startswith("crate_count_")]
features_to_keep += [col for col in df_train_subset.columns if col.startswith("article_id_")]

# Filtered subset dataframe
df_train_filtered = df_train_subset[features_to_keep]
df_test_filtered = df_test[features_to_keep]

# Update X_train and X_test
X_train = df_train_filtered.astype(float).values
y_train = df_train_subset[target].astype(float).values
X_test = df_test_filtered.astype(float).values
y_test = df_test[target].astype(float).values

# Now continue to scaling and modeling...
# Skalieren der Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konvertiere die Daten in Torch-Tensoren
train_x = torch.from_numpy(X_train_scaled).float()
train_y = torch.from_numpy(y_train).float()
test_x = torch.from_numpy(X_test_scaled).float()
test_y = torch.from_numpy(y_test).float()

# Flatten target explicitly before training
train_y = train_y.flatten()

################################################################################
# Definition des Deep GP Modells (2 Schichten) mittels GPyTorch
################################################################################

class DeepGPHiddenLayer(AbstractDeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128):
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DeepGPOutputLayer(AbstractDeepGPLayer):
    def __init__(self, input_dims, num_inducing=128):
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims=1)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DeepGPModel(DeepGP):
    def __init__(self, input_dim):
        super().__init__()
        # first layer reduces dimension to 2 (arbitrary small number)
        self.hidden_layer = DeepGPHiddenLayer(input_dims=input_dim, output_dims=2, num_inducing=64)
        # second (output) layer outputs exactly 1 dimension
        self.output_layer = DeepGPOutputLayer(input_dims=2, num_inducing=64)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        hidden_output = self.hidden_layer(x)
        output = self.output_layer(hidden_output.mean)
        return output

################################################################################
# Training und Evaluation des Deep GP Modells
################################################################################
def deep_gp_regression():
    train_y_flat = train_y.flatten()

    model = DeepGPModel(input_dim=train_x.shape[-1])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': likelihood.parameters()}], lr=0.01)

    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))
    )

    training_iter = 100
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_flat)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'Iteration {i+1}/{training_iter} - Loss: {loss.item():.3f}')

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        y_pred = preds.mean.cpu().numpy()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    confidence_interval = confidence_interval(y_pred)

    print("Deep GP Regression Test Results:")
    print(f"MSE: {mse}, MAE: {mae}, R2: {r2}, Confidence Interval: {confidence_interval}")

################################################################################
# Main-Block
################################################################################
if __name__ == '__main__':
    deep_gp_regression()