import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
from _1_Preprocessing import load_data_from_parquet, subsample_for_plotting

# Load and preprocess data
df_train, _ = load_data_from_parquet()
df_train = subsample_for_plotting(df_train)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Service Time Prediction Dashboard", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='correlation-heatmap'),
            dcc.Graph(id='target-distribution'),
            dcc.Graph(id='service-time-start-histogram')
        ], width=12, lg=8),
        dbc.Col([
            html.H3("Input New Sample"),
            dbc.FormGroup([
                dbc.Label("Article Weight (g)"),
                dbc.Input(id='input-article-weight', type='number', value=500),
                dbc.Label("Warehouse ID"),
                dbc.Input(id='input-warehouse-id', type='number', value=1),
                dbc.Label("Driver ID"),
                dbc.Input(id='input-driver-id', type='number', value=1),
                dbc.Label("Is Business"),
                dbc.Input(id='input-is-business', type='number', value=0),
                dbc.Label("Is Pre Order"),
                dbc.Input(id='input-is-pre-order', type='number', value=0),
                dbc.Label("Has Elevator"),
                dbc.Input(id='input-has-elevator', type='number', value=1),
                dbc.Label("Floor"),
                dbc.Input(id='input-floor', type='number', value=1),
                dbc.Button("Predict", id='predict-button', color='success', className="mt-3")
            ]),
            html.Div(id='prediction-output')
        ], width=12, lg=4)
    ])
], fluid=True, style={'backgroundColor': '#7EB53C'})

# Define callbacks
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Output('target-distribution', 'figure'),
    Output('service-time-start-histogram', 'figure'),
    Input('correlation-heatmap', 'id')
)
def update_graphs(_):
    # Correlation Heatmap
    corr = df_train[["floor", "article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "num_previous_orders_customer", "customer_speed", "service_time_start"]].corr()
    np.fill_diagonal(corr.values, np.nan)
    heatmap = px.imshow(corr, color_continuous_scale='Greens', title='Feature Correlation')

    # Target Distribution
    target_dist = px.histogram(df_train, x='service_time_in_minutes', nbins=50, title='Distribution of Service Time', color_discrete_sequence=['#7EB53C'])

    # Service Time Start Histogram
    service_time_hist = px.histogram(df_train, x='service_time_start', nbins=24, title='Service Time Start Distribution', color_discrete_sequence=['#7EB53C'])

    return heatmap, target_dist, service_time_hist

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-article-weight', 'value'),
    State('input-warehouse-id', 'value'),
    State('input-driver-id', 'value'),
    State('input-is-business', 'value'),
    State('input-is-pre-order', 'value'),
    State('input-has-elevator', 'value'),
    State('input-floor', 'value')
)
def predict(n_clicks, article_weight, warehouse_id, driver_id, is_business, is_pre_order, has_elevator, floor):
    if n_clicks is None:
        return ""

    # Prepare the input sample
    sample = pd.DataFrame([{
        "article_weight_in_g": article_weight,
        "warehouse_id": warehouse_id,
        "driver_id": driver_id,
        "is_business": is_business,
        "is_pre_order": is_pre_order,
        "has_elevator": has_elevator,
        "floor": floor
    }])

    # Load models from disk
    lr_model = joblib.load('./model/linear_regression.pkl')
    xgb_model = joblib.load('./model/xgboost.pkl')

    # Make predictions
    lr_pred = lr_model.predict(sample)[0]
    xgb_pred = xgb_model.predict(sample)[0]

    # Confidence intervals (dummy values for illustration)
    lr_conf = (lr_pred - 5, lr_pred + 5) # TODO implement correct conf intervals
    xgb_conf = (xgb_pred - 5, xgb_pred + 5)

    return html.Div([
        html.H4("Predictions:"),
        html.P(f"Linear Regression: {lr_pred:.2f} (95% CI: {lr_conf[0]:.2f} - {lr_conf[1]:.2f})"),
        html.P(f"XGBoost: {xgb_pred:.2f} (95% CI: {xgb_conf[0]::.2f} - {xgb_conf[1]:.2f})"),
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)