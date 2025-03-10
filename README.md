# STADS Datathon 2025 - Flaschenpost Challenge

## Overview
This project is the result of our participation in the Flaschenpost Challenge during the STADS Datathon 2025. Our goal was to develop a prediction model for the customer service time of the German company Flaschenpost.

## Approach
We explored various models and found that Bayesian Models were particularly suitable for our needs. Our workflow included the following steps:

1. **Data Preprocessing**: We cleaned and prepared our data in the `_1_Preprocessing` directory.
2. **Model Implementation**: We implemented different models in directories `_2_` through `_11_`.
3. **Model Evaluation**: We evaluated the performance of our models using a dashboard located in the `_99_dash` directory.

## Models
A brief explanation of the models we used:

- **Baseline**: A simple model used as a reference point for evaluating the performance of other models. It always predicts the average service time.
- **Linear Regression (_2_LinearRegression.py)**: A basic predictive model that assumes a linear relationship between the input features and the target variable.
- **XGBoost (_3_XGBoost.py)**: An optimized gradient boosting algorithm.
- **Neuronal Network (_4_NeuronalesNetz.py)**: A model consisting of layers of interconnected nodes.
- **Bayesian Ridge Regression (_5_BayesRidgeRegression.py)**: A linear regression model that incorporates Bayesian inference, providing probabilistic predictions and regularization.
- **BART (_7_BART.py)**: Bayesian Additive Regression Trees, a non-parametric model that combines the strengths of decision trees and Bayesian inference.
- **LinexXGB (_8_LinexXGB.py)**: A combination of linear regression and XGBoost, leveraging the strengths of both models. Here we incorporated a personalized cost function.
- **LightGBM (_9_LightGBM.py)**: A gradient boosting framework that uses tree-based learning algorithms.
- **Deep Gaussian Processes (_10_DeepGP.py)**: A model that extends Gaussian processes to deep architectures.
- **Hierarchical Bayesian Models (_11_HierarchicalBayes.py)**: Models that incorporate hierarchical structures, allowing for more flexible and accurate predictions by sharing information across different levels of the hierarchy.

## Results
We significantly improved the prediction accuracy by reducing the Mean Absolute Error (MAE) from 4.330 minutes with the Baseline model to 2.316 minutes with the XGBoost model. This represents a reduction of approximately 46.5% in the prediction error. Additionally, the XGBoost model achieved a Mean Squared Error (MSE) of 11.299 and an R-squared (R2) value of 0.769, indicating a strong fit to the data. The confidence interval for the XGBoost model was (-9.759, 9.787), showing a narrower range compared to the Baseline model's interval of (-9.801, 9.801), further demonstrating the improved precision of our predictions.

## Final Presentation
Our final presentation, summarizing our results and findings, is available in the `pitch_summary` directory. With this pitch, we secured second place in the challenge.

## Repository Structure
- `_0_...`: Exploratory Data Analysis
- `_1_Preprocessing`: Data preprocessing scripts and notebooks.
- `_2_` to `_11_`: Implementation of various prediction models.
- `_12_evaluation`: Definition of confidence interval calculation.
- `_99_dash`: Dashboard for model evaluation.
- `pitch_summary`: Final presentation of our results.

## Team Members
- [Paul König](https://github.com/p-koenig)
- [Emil Schallwig](https://github.com/limescha22)
- [Simon Schumacher](https://github.com/SparklingCraft)
- [Elise Wolf](https://github.com/eelisee)
