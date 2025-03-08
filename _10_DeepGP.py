import pandas as pd
import numpy as np
import torch
import gpytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from _1_Preprocessing import run_preprocessing

# Zielvariable und Features definieren
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

# Konvertiere die Daten in Torch-Tensoren
train_x = torch.from_numpy(X_train_scaled).float()
train_y = torch.from_numpy(y_train).float()
test_x = torch.from_numpy(X_test_scaled).float()
test_y = torch.from_numpy(y_test).float()

################################################################################
# Definition des Deep GP Modells (2 Schichten) mittels GPyTorch
################################################################################

# Erste (versteckte) Schicht: Ein DeepGPLayer
class DeepGPHiddenLayer(gpytorch.models.DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128):
        # Inducing Points initialisieren (zufällig)
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

# Zweite (letzte) Schicht: Ebenfalls ein DeepGPLayer, aber als Ausgabe mit 1 Dimension
class DeepGPOutputLayer(gpytorch.models.DeepGPLayer):
    def __init__(self, input_dims, num_inducing=128):
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, 1)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Gesamtmodell: Zwei Schichten Deep GP
class DeepGPModel(gpytorch.models.DeepGP):
    def __init__(self, train_x_shape):
        super().__init__()
        # Versteckte Schicht: Eingabe-Dimension entspricht der Feature-Anzahl, Ausgabe z. B. 2
        self.hidden_layer = DeepGPHiddenLayer(input_dims=train_x_shape[-1], output_dims=2, num_inducing=128)
        # Ausgabe-Schicht: Eingangsdimension 2, Ausgabe 1
        self.output_layer = DeepGPOutputLayer(input_dims=2, num_inducing=128)
        # Likelihood: Gaussian
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
    def forward(self, x):
        hidden_output = self.hidden_layer(x)
        # Verwende als Input für die letzte Schicht den Mittelwert der Verteilung der versteckten Schicht
        output = self.output_layer(hidden_output.mean)
        return output
    
    # Optional: Eine Hilfsmethode für Vorhersagen
    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.forward(x))
        return preds.mean.squeeze()

################################################################################
# Training und Evaluation des Deep GP Modells
################################################################################

def deep_gp_regression():
    ########################################################################################################################
    # Deep GP Regression: Training und Evaluation
    print("Training Deep Gaussian Process model...")
    
    model = DeepGPModel(train_x.shape)
    model.train()
    model.likelihood.train()
    
    # Verwende den Adam-Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Verwende die DeepApproximateMLL als Loss
    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_x.size(0))
    )
    
    training_iter = 500
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if (i+1) % 50 == 0:
            print(f"Iteration {i+1}/{training_iter} - Loss: {loss.item():.3f}")
        optimizer.step()
    
    # Evaluation auf Testdaten
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = model.likelihood(model(test_x))
    y_pred = preds.mean.squeeze().numpy()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Deep GP Regression Evaluation on Test-Set:")
    print(f"MSE = {mse}")
    print(f"MAE = {mae}")
    print(f"R2  = {r2}")
    
    return model, y_pred

################################################################################
# Main-Block
################################################################################
if __name__ == '__main__':
    deep_gp_regression()