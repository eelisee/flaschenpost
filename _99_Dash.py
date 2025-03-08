import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, State

from _1_Preprocessing import load_data_from_parquet, subsample_for_plotting

# Load and preprocess data
df_train, _ = load_data_from_parquet()
df_train = subsample_for_plotting(df_train)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Service Time Dashboard", className="mb-4",
                        style={"color": "white", "textAlign": "left", "paddingTop": "10px"}), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='correlation-heatmap'),
            dcc.Graph(id='target-distribution', style={"marginTop": "-20px"}),
        ], width=12, lg=8),
        dbc.Col([
            html.H3("Model Playground"),
            dbc.FormGroup([
                dbc.Label("Article Weight (g)", className="pt-2", style={"marginTop": "10px"}),
                dbc.Input(id='input-article-weight', type='number', value=5000),
                dbc.Label("Warehouse ID", className="pt-2", style={"marginTop": "10px"}),
                dbc.Input(id='input-warehouse-id', type='number', value=1),
                # dbc.Label("Driver ID", className="pt-2", style={"marginTop": "10px"}),
                # dbc.Input(id='input-driver-id', type='number', value=1),
                dbc.Label("Business Customer", className="pt-2", style={"marginTop": "10px"}),
                dbc.Label("Customer Experience", className="pt-2", style={"marginTop": "-10px"}),
                dbc.Input(id='input-is-pre-order', type='number', value=2),
                dbc.Label("Floor", className="pt-2", style={"marginTop": "10px"}),
                dbc.Input(id='input-floor', type='number', value=1),
                dbc.Row([
                    dbc.Col(dbc.FormGroup([
                        dbc.Label("Business Customer", className="pt-2", style={"marginTop": "-10px"}),
                        dbc.Checkbox(id='input-is-business', checked=False),
                    ]), width=4),
                    dbc.Col(dbc.FormGroup([
                        dbc.Label("Pre Order", className="pt-2", style={"marginTop": "-10px"}),
                        dbc.Checkbox(id='input-is-pre-order', checked=False),
                    ]), width=4),
                    dbc.Col(dbc.FormGroup([
                        dbc.Label("Elevator available", className="pt-2", style={"marginTop": "-10px"}),
                        dbc.Checkbox(id='input-has-elevator', checked=False, style={"marginleft": "-10px"}),
                    ]), width=4),
                ], className="mb-3"),
                dbc.Button("Predict", id='predict-button', color='success', className="mt-3"),
            ]),
            html.Div(id='prediction-output')
        ], width=12, lg=4)
    ])
], fluid=True, style={'backgroundColor': '#7EB53C'})
# Define the layout


# Define callbacks
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Output('target-distribution', 'figure'),
    # Output('service-time-start-histogram', 'figure'),
    Input('correlation-heatmap', 'id')
)
def update_graphs(_):
    # Correlation Heatmap
    corr = df_train[
        ["floor", "article_weight_in_g", "is_business", "is_pre_order", "has_elevator", "num_previous_orders_customer",
         "customer_speed", "service_time_start"]].corr()
    np.fill_diagonal(corr.values, np.nan)
    corr.columns = ['Floor', 'Article Weight (g)', 'Business Customer', 'Pre Order', 'Elevator available',
                    'Customer Experience', 'Customer Speed (avg.)', 'Time of day']
    corr.index = ['Floor', 'Article Weight (g)', 'Business Customer', 'Pre Order', 'Elevator available',
                  'Customer Experience', 'Customer Speed (avg.)', 'Time of day']
    heatmap = px.imshow(corr, color_continuous_scale='Greens', title='Feature Correlation', width=1050, height=500, aspect='auto',
                        text_auto='.2f')
    heatmap.update_layout(
        margin=dict(l=60, r=60, t=50, b=50),
        paper_bgcolor='#7EB53C',
        plot_bgcolor='#7EB53C',
        font=dict(color='white', size=16)
    )

    # Target Distribution
    target_dist = px.histogram(df_train, x='service_time_in_minutes', nbins=50, title='Distribution of Service Time',
                               color_discrete_sequence=['white'], width=1050, height=400, labels={'service_time_in_minutes': 'Service Time (Minutes)'})
    target_dist.update_layout(
        margin=dict(l=60, r=60, t=50, b=50),
        paper_bgcolor='#7EB53C',
        plot_bgcolor='#7EB53C',
        font=dict(color='white', size=16)
    )

    return heatmap, target_dist


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-article-weight', 'value'),
    State('input-warehouse-id', 'value'),
    # State('input-driver-id', 'value'),
    State('input-is-business', 'checked'),
    State('input-is-pre-order', 'checked'),
    State('input-has-elevator', 'checked'),
    State('input-floor', 'value')
)
def predict(n_clicks, article_weight, warehouse_id, is_business, is_pre_order, has_elevator, floor):
    if n_clicks is None:
        return ""

    is_business = int(is_business)
    is_pre_order = int(is_pre_order)
    has_elevator = int(has_elevator)

    # Prepare the input sample
    sample = pd.DataFrame([{
        "article_weight_in_g": article_weight,
        "is_business": is_business,
        "is_pre_order": is_pre_order,
        "has_elevator": has_elevator,
        "floor": floor,
        "num_previous_orders_customer": 0,
        "customer_speed": 4,
        "service_time_start": 12,
    }])

    # Load models from disk
    lr_model = joblib.load('./model/linear_regression.pkl')
    xgb_model = joblib.load('./model/xgboost.pkl')

    # Make predictions
    lr_pred = lr_model.predict(sample)[0]
    xgb_pred = xgb_model.predict(sample)[0]

    # Confidence intervals (dummy values for illustration)
    lr_conf = (lr_pred - 5, lr_pred + 5)  # TODO implement correct conf intervals
    xgb_conf = (xgb_pred - 5, xgb_pred + 5)

    # return html.Div([
    #     html.H4("Predictions:"),
    #     html.P(f"Linear Regression: {round(lr_pred, 2)} (95% CI: {round(lr_conf[0], 2)} - {round(lr_conf[1], 2)})"),
    #     html.P(f"XGBoost: {round(xgb_pred, 2)} (95% CI: {round(xgb_conf[0], 2)} - {round(xgb_conf[1], 2)})"),
    # ])
    return dbc.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("Y (Minutes)"), html.Th("CI")])),
        html.Tbody([
            html.Tr([html.Td("Linear Regression"), html.Td(round(lr_pred, 2)),
                     html.Td(f"{round(lr_conf[0], 2)} - {round(lr_conf[1], 2)}")]),
            html.Tr([html.Td("XGBoost"), html.Td(round(xgb_pred, 2)),
                     html.Td(f"{round(xgb_conf[0], 2)} - {round(xgb_conf[1], 2)}")])
        ])
    ], bordered=True, hover=True, responsive=True, striped=True)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
