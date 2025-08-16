# app.py
# FINAL SCRIPT (With Stable EDA Sub-Tabs & Mobile Fixes)

import os
import re
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from catboost import CatBoostRegressor
import base64
import io

import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
MODELS_ROOT = "models"
DATA_PATH = "data/original_data.csv"
PROPERTY_FOLDERS = [f"BlendProperty{i}" for i in range(1, 11)]
PROPERTIES = [f"Property {i+1}" for i in range(10)]
COMPONENTS = [f"Component {i+1}" for i in range(5)]
APP_TITLE = "🚀 Fuel Blend Ensemble Predictor"

# ========== ANN MODEL ARCHITECTURE ==========
class DynamicANN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, p1=0.5, p2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.BatchNorm1d(hidden1), nn.Dropout(p1),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(p2),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x): return self.net(x)

def _find_linear_shapes_from_state_dict(sd: dict):
    def _get_layer_num(key):
        match = re.search(r'\.(\d+)\.weight$', key)
        return int(match.group(1)) if match else 999
    linear_weights = sorted(
        [(k, tuple(v.shape)) for k, v in sd.items() if k.endswith(".weight") and getattr(v, "ndim", 0) == 2],
        key=lambda kv: _get_layer_num(kv[0])
    )
    if len(linear_weights) < 3: raise ValueError(f"Expected ≥3 Linear weights, found {len(linear_weights)}")
    h1, in_dim = linear_weights[0][1]
    h2, _ = linear_weights[1][1]
    out, _ = linear_weights[2][1]
    if out != 1: raise ValueError("ANN output dimension not 1")
    return in_dim, h1, h2

def load_ann_model(pt_path):
    chk = torch.load(pt_path, map_location="cpu")
    sd = chk.get("state_dict", chk) if isinstance(chk, dict) else chk
    if sd is None: raise ValueError("Unrecognized ANN checkpoint format")
    in_dim, h1, h2 = _find_linear_shapes_from_state_dict(sd)
    model = DynamicANN(in_dim, h1, h2)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

# ========== ENSEMBLE PREDICTOR CLASS ==========
class PropertyEnsemble:
    def __init__(self, folder):
        self.folder = folder
        self.x_scaler = joblib.load(os.path.join(folder, "x_scaler.pkl"))
        self.y_scaler = joblib.load(os.path.join(folder, "y_scaler.pkl"))
        models_to_load = [("xgb", "xgb.pkl"), ("lgb", "lgb.pkl"), ("svr", "svr.pkl"), ("lin", "lin.pkl")]
        self.models = {name: joblib.load(os.path.join(folder, fname)) for name, fname in models_to_load}
        self.models["cat"] = CatBoostRegressor().load_model(os.path.join(folder, "cat.cbm"))
        self.meta = joblib.load(os.path.join(folder, "meta.pkl"))
        self.ann = load_ann_model(os.path.join(folder, "ann.pt"))
        self.input_cols = list(getattr(self.x_scaler, "feature_names_in_", []))

    def _align_df(self, df: pd.DataFrame):
        if not self.input_cols: return df
        df_aligned = df.copy()
        for col in self.input_cols:
            if col not in df_aligned.columns: df_aligned[col] = 0
        return df_aligned[self.input_cols]

    def predict(self, df: pd.DataFrame):
        X_aligned = self._align_df(df)
        X_scaled = self.x_scaler.transform(X_aligned.values)
        with torch.no_grad():
            ann_out_scaled = self.ann(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
        base_preds_scaled = [ann_out_scaled]
        for name in ["xgb", "lgb", "cat", "svr", "lin"]:
            raw = self.models[name].predict(X_scaled)
            base_preds_scaled.append(raw.reshape(-1, 1) if raw.ndim == 1 else raw)
        stacked_unscaled = np.hstack([self.y_scaler.inverse_transform(p) for p in base_preds_scaled])
        final = self.meta.predict(stacked_unscaled).reshape(-1, 1)
        pred_std = np.std(stacked_unscaled, axis=1, keepdims=True)
        return final, pred_std, stacked_unscaled

# ========== GLOBAL DATA LOADING ==========
ENSEMBLES = {}
for prop in PROPERTY_FOLDERS:
    folder_path = os.path.join(MODELS_ROOT, prop)
    if os.path.isdir(folder_path):
        try:
            ENSEMBLES[prop] = PropertyEnsemble(folder_path)
        except Exception as e:
            print(f"Failed to load ensemble for {prop}: {e}")
try:
    DF_EDA = pd.read_csv(DATA_PATH)
except Exception:
    DF_EDA = pd.DataFrame()

# ========== DASH APPLICATION ==========
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.title = APP_TITLE

custom_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccd6f6"),
        title_font=dict(color="#ccd6f6"),
        legend_font=dict(color="#ccd6f6"),
        xaxis={"gridcolor": "rgba(100, 255, 218, 0.1)", "linecolor": "rgba(100, 255, 218, 0.2)"},
        yaxis={"gridcolor": "rgba(100, 255, 218, 0.1)", "linecolor": "rgba(100, 255, 218, 0.2)"}
    )
)

# ========== HELPER FUNCTIONS AND LAYOUT COMPONENTS ==========
PIPELINE_STEPS_LAYMAN = [
    {"title": "Step 1: The Raw Recipe.", "text": "You provide the 'recipe' for your fuel blend by entering the concentration and properties of each component. This is the basic input for the team.", "image": "tech_step1_input.png", "icon": "bi bi-keyboard-fill"},
    {"title": "Step 2: The Data Sleuths.", "text": "Before handing the recipe to the experts, the system's 'data detectives' analyze it from every angle. They calculate new, hidden insights, such as the **average** of all the properties and how much they **vary** from one another. This helps the experts understand not just what's in the blend, but how the ingredients interact.", "image": "tech_step2_features.png", "icon": "bi bi-gear-fill"},
    {"title": "Step 3: The Expert Panel.", "text": "The enhanced recipe is given to a diverse panel of experts, including a deep-learning specialist (the ANN), a master of logic trees (XGBoost/CatBoost), and a statistician (Linear/SVR). Each expert makes their own prediction independently.", "image": "tech_step3_base_models.png", "icon": "bi bi-people-fill"},
    {"title": "Step 4: The Manager's Meeting.", "text": "The predictions from all the experts are not taken at face value. Instead, they are given to a 'manager' model. The manager is a highly-trained expert in its own right, whose sole job is to evaluate the other experts. It learns when to trust one expert more than another and how to combine their opinions to get the most accurate answer.", "image": "tech_step4_meta_model.png", "icon": "bi bi-person-check-fill"},
    {"title": "Step 5: The Final Verdict.", "text": "The manager weighs all the opinions and delivers the single, definitive prediction for the fuel blend property. This is a much more reliable result than relying on any single expert, as it leverages the collective intelligence of the entire team.", "image": "tech_step5_prediction.png", "icon": "bi bi-trophy-fill"},
]
PIPELINE_STEPS_TECHNICAL = [
    {"title": "Step 1 & 2: Raw Data & Feature Engineering", "text": "The system starts with a `5x10` matrix of component properties and a `1x5` vector of component concentrations. This raw input is transformed into a richer feature set, including weighted averages and statistical features like `mean` and `standard deviation`. This enriched feature set is then standardized using a `StandardScaler`.", "image": "tech_step1_feature_engineering.png", "icon": "bi bi-keyboard-fill"},
    {"title": "Step 3: Base Learner Predictions (Level-0 Models)", "text": "The standardized features are fed into a diverse collection of base models, including a **Neural Network (ANN)**, **XGBoost**, **LightGBM**, **CatBoost**, **SVR**, and **Linear Regression**. Each model outputs its own prediction for the target property.", "image": "tech_step2_base_learners.png", "icon": "bi bi-people-fill"},
    {"title": "Step 4: Stacking & Meta-Model (Level-1 Model)", "text": "The predictions from the base learners are 'stacked' together to form a new input vector. This vector is fed to a final **meta-model** trained to find the optimal way to combine the base predictions, correcting for their individual biases and variances.", "image": "tech_step3_stacking_meta_model.png", "icon": "bi bi-person-check-fill"},
    {"title": "Step 5: Final Ensemble Prediction", "text": "The output from the meta-model is the final, definitive prediction. This stacked generalization approach is a powerful technique for improving predictive performance by leveraging the strengths of multiple models.", "image": "tech_step4_final_prediction.png", "icon": "bi bi-trophy-fill"},
]

def create_pipeline_step_card(step_data):
    image_path = os.path.join("assets", step_data["image"])
    if os.path.exists(image_path):
        image_component = html.Img(src=app.get_asset_url(step_data["image"]), className="img-fluid rounded")
    else:
        icon_class = step_data["icon"] + " text-muted"
        image_component = html.Div(html.I(className=icon_class), className="pipeline-icon-fallback")
    return dbc.Row([
        dbc.Col(dbc.Card(image_component, body=True), lg=5, className="mb-3 mb-lg-0"),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(step_data["title"]),
            dcc.Markdown(step_data["text"], className="lead"),
        ]), className="h-100"), lg=7),
    ], align="stretch")

def make_input_card(comp_idx, comp_name):
    return dbc.AccordionItem(
        title=f"🔹 {comp_name}",
        item_id=f"item-{comp_idx}",
        children=[
            dbc.Label("Concentration (%)"),
            dbc.Row([
                dbc.Col(
                    dcc.Slider(
                        min=0, max=100, step=1, value=20 if comp_idx == 0 else 0,
                        marks=None, tooltip={"placement": "bottom", "always_visible": False},
                        id={'type': 'percentage-slider', 'index': comp_idx}
                    ), width=8, sm=9
                ),
                dbc.Col(
                    dbc.InputGroup([
                        dbc.Input(
                            type="number", min=0, max=100, step=1,
                            value=20 if comp_idx == 0 else 0,
                            id={'type': 'percentage-input', 'index': comp_idx}
                        ),
                        dbc.InputGroupText("%")
                    ]), width=4, sm=3
                ),
            ], align="center"),
            html.Hr(),
            dbc.Row([
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText(f"P{prop_idx+1}"),
                    dbc.Input(type="number", value=0, step=0.01, id={'type': 'prop-input', 'index': f"{comp_idx}-{prop_idx}"})
                ]), md=6) for prop_idx in range(len(PROPERTIES))
            ], className="g-2")
        ]
    )

def parse_contents_headerless(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
        return dbc.Alert("This is not a CSV file!", color="danger")
    except Exception as e:
        return dbc.Alert(f'There was an error processing this file: {e}', color="danger")

# Organize EDA dropdown options
blend_properties = sorted([c for c in DF_EDA.columns if "BlendProperty" in c])
component_fractions = sorted([c for c in DF_EDA.columns if "fraction" in c])
component_properties = sorted([c for c in DF_EDA.columns if "Component" in c and "Property" in c])
eda_dropdown_options = (
    [{"label": c, "value": c} for c in blend_properties] +
    [{"label": c, "value": c} for c in component_fractions] +
    [{"label": c, "value": c} for c in component_properties]
)

# ========== APP LAYOUT ==========
tab_style = {'background-color': '#08111a', 'color': '#f0f8ff', 'border': 'none', 'border-bottom': '2px solid transparent', 'border-radius': '12px 12px 0 0', 'padding': '12px 20px', 'margin-right': '4px'}
active_tab_style = {'background-color': '#123456', 'color': '#ffffff', 'font-weight': 'bold', 'border-bottom': '2px solid #64ffda'}
datatable_style = {'style_as_list_view': True, 'style_cell': {'textAlign': 'left', 'padding': '5px', 'backgroundColor': '#1a324a', 'color': '#FFFFFF', 'border': '1px solid #254a6b'}, 'style_header': {'backgroundColor': '#254a6b', 'color': 'white', 'fontWeight': 'bold'}}
upload_style = {'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '10px'}

# --- RE-ARCHITECTED: EDA Tab with Sub-Tabs for stability ---
eda_tab_content = dbc.Card(
    [
        dbc.CardHeader(
            dbc.Tabs(
                [
                    dbc.Tab(label="Correlation", tab_id="eda-tab-corr"),
                    dbc.Tab(label="Distribution (Histogram)", tab_id="eda-tab-hist"),
                    dbc.Tab(label="Relationship (Scatter)", tab_id="eda-tab-scatter"),
                ],
                id="eda-sub-tabs",
                active_tab="eda-tab-corr",
            )
        ),
        dbc.CardBody(html.Div(id="eda-sub-tab-content")),
    ],
    className="mt-4"
)

app.layout = html.Div([
    dcc.Store(id='pipeline-step-store-layman', data=0),
    dcc.Store(id='pipeline-step-store-technical', data=0),
    dbc.NavbarSimple(brand=APP_TITLE, color="primary", dark=True, className="mb-4"),
    dbc.Container([
        dbc.Tabs(id="main-tabs", active_tab="tab-workbench", children=[
            dbc.Tab(label="Prediction Workbench", tab_id="tab-workbench", tab_style=tab_style, active_tab_style=active_tab_style, children=dbc.Row([
                dbc.Col(dbc.Card(className="fade-in", children=[
                    dbc.CardHeader(html.H4("🧪 Input Workbench")),
                    dbc.CardBody([
                        dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop a CSV file or ', html.A('Select File')]), style=upload_style, multiple=False),
                        html.Div(id='upload-message', className='mb-3'),
                        dbc.Accordion([make_input_card(i, name) for i, name in enumerate(COMPONENTS)], always_open=True, active_item="item-0"),
                        dbc.Button([html.I(className="bi bi-robot me-2"), "Run Predictions"], id='predict-button', n_clicks=0, className="w-100 mt-4", size="lg", color="primary")
                    ])
                ]), lg=6, className="mb-4"),
                dbc.Col(dcc.Loading(children=dbc.Row([
                    dbc.Col(dbc.Card(dcc.Graph(id='pie-chart'), className="fade-in"), width=12, className="mb-4"),
                    dbc.Col(dbc.Card(id="prediction-output-card", body=True, children="Click 'Run Predictions' to see results."), width=12),
                ])), lg=6),
            ], className="mt-4")),
            dbc.Tab(label="Model Details", tab_id="tab-model-details", tab_style=tab_style, active_tab_style=active_tab_style, children=dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("Ensemble and Network Architecture", className="card-title"),
                html.P("Details of the underlying models for a selected property."),
                dcc.Dropdown(id="model-details-dropdown", options=[{'label': k, 'value': k} for k in ENSEMBLES.keys()], value=list(ENSEMBLES.keys())[0] if ENSEMBLES else None, clearable=False, className="mb-3"),
                dcc.Loading(id="model-details-output")
            ])), width=12)], className="mt-4")),
            dbc.Tab(label="Layman's Intuition", tab_id="tab-layman-intuition", tab_style=tab_style, active_tab_style=active_tab_style, children=html.Div([
                html.H2("The Prediction Pipeline: A Simple Guide", className="text-center my-4"),
                dbc.Card(create_pipeline_step_card(PIPELINE_STEPS_LAYMAN[0]), id="pipeline-display-card-layman", body=True),
                dbc.Progress(id="pipeline-progress-layman", value=1, max=len(PIPELINE_STEPS_LAYMAN), className="my-4"),
                dbc.Row([dbc.Col(dbc.Button("Previous", id="prev-step-button-layman", outline=True, color="info", className="w-100")), dbc.Col(dbc.Button("Next", id="next-step-button-layman", outline=True, color="info", className="w-100"))], justify="between")
            ], className="mt-4")),
            dbc.Tab(label="Technical Overview", tab_id="tab-technical-overview", tab_style=tab_style, active_tab_style=active_tab_style, children=html.Div([
                html.H2("The Prediction Pipeline: Technical Details", className="text-center my-4"),
                dbc.Card(create_pipeline_step_card(PIPELINE_STEPS_TECHNICAL[0]), id="pipeline-display-card-technical", body=True),
                dbc.Progress(id="pipeline-progress-technical", value=1, max=len(PIPELINE_STEPS_TECHNICAL), className="my-4"),
                dbc.Row([dbc.Col(dbc.Button("Previous", id="prev-step-button-technical", outline=True, color="info", className="w-100")), dbc.Col(dbc.Button("Next", id="next-step-button-technical", outline=True, color="info", className="w-100"))], justify="between")
            ], className="mt-4")),
            dbc.Tab(label="Exploratory Data Analysis", tab_id="tab-eda", tab_style=tab_style, active_tab_style=active_tab_style, children=eda_tab_content),
        ])
    ], fluid=True),
])

# ========== CALLBACKS ==========
@app.callback(
    Output({'type': 'percentage-slider', 'index': ALL}, 'value'),
    Output({'type': 'percentage-input', 'index': ALL}, 'value'),
    Input({'type': 'percentage-slider', 'index': ALL}, 'value'),
    Input({'type': 'percentage-input', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def sync_concentration_inputs(slider_vals, input_vals):
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return no_update, no_update
    slider_output = [no_update] * len(slider_vals)
    input_output = [no_update] * len(input_vals)
    idx = triggered_id['index']
    
    if triggered_id['type'] == 'percentage-slider':
        new_value = slider_vals[idx]
        if new_value is not None:
            input_output[idx] = new_value
    elif triggered_id['type'] == 'percentage-input':
        new_value = input_vals[idx]
        if new_value is not None:
            slider_output[idx] = new_value
    return slider_output, input_output

@app.callback(
    Output({'type': 'percentage-slider', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'percentage-input', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'prop-input', 'index': ALL}, 'value'),
    Output('upload-message', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_inputs_from_upload(contents, filename):
    if contents is None:
        return no_update, no_update, no_update, ""
    df = parse_contents_headerless(contents, filename)
    if not isinstance(df, pd.DataFrame):
        return no_update, no_update, no_update, df
    if len(df) != 1 or len(df.columns) < 55:
        error_msg = dbc.Alert("Error: CSV must have one row with at least 55 columns.", color="danger")
        return no_update, no_update, no_update, error_msg
    try:
        single_row = df.iloc[0]
        slider_values = [single_row[i] for i in range(5)]
        prop_values = [float(single_row[i]) for i in range(5, 55)]
        message = dbc.Alert(f"Successfully loaded headerless data from {filename}", color="success")
        return slider_values, slider_values, prop_values, message
    except Exception as e:
        error_msg = dbc.Alert(f"Error processing data by position: {e}", color="danger")
        return no_update, no_update, no_update, error_msg

# --- NEW: Callback to render the content of the active EDA sub-tab ---
@app.callback(
    Output("eda-sub-tab-content", "children"),
    Input("eda-sub-tabs", "active_tab")
)
def render_eda_tab_content(active_tab):
    if active_tab == "eda-tab-corr":
        return dbc.Row([
            dbc.Col(dcc.Checklist(id="eda-corr-checklist", options=blend_properties + component_fractions, value=blend_properties, labelStyle={"display": "block", "margin-bottom": "5px"}), width=12, lg=3),
            dbc.Col(dcc.Loading(dcc.Graph(id="eda-corr-graph", style={"height": "75vh"})), width=12, lg=9)
        ])
    elif active_tab == "eda-tab-hist":
        return dbc.Row([
            dbc.Col([
                dbc.Label("Variable"),
                dcc.Dropdown(id="eda-hist-x", options=eda_dropdown_options),
                dbc.Label("Number of Bins", className="mt-3"),
                dcc.Slider(id="eda-hist-bins", min=10, max=100, step=10, value=30, marks=None, tooltip={"placement": "bottom", "always_visible": True})
            ], width=12, lg=3),
            dbc.Col(dcc.Loading(dcc.Graph(id="eda-hist-graph", style={"height": "75vh"})), width=12, lg=9)
        ])
    elif active_tab == "eda-tab-scatter":
        return dbc.Row([
            dbc.Col([
                dbc.Label("X-Axis"),
                dcc.Dropdown(id="eda-scatter-x", options=eda_dropdown_options),
                dbc.Label("Y-Axis", className="mt-3"),
                dcc.Dropdown(id="eda-scatter-y", options=eda_dropdown_options),
                dbc.Label("Color (Optional)", className="mt-3"),
                dcc.Dropdown(id="eda-scatter-color", options=eda_dropdown_options),
            ], width=12, lg=3),
            dbc.Col(dcc.Loading(dcc.Graph(id="eda-scatter-graph", style={"height": "75vh"})), width=12, lg=9)
        ])
    return html.P("This tab has no content.")

# --- NEW: Separate callbacks for each EDA plot ---
@app.callback(
    Output("eda-corr-graph", "figure"),
    Input("eda-corr-checklist", "value")
)
def update_eda_heatmap(corr_vars):
    if not corr_vars:
        return go.Figure().update_layout(title_text="Please select variables for the heatmap", template=custom_template)
    corr_df = DF_EDA[corr_vars].corr(numeric_only=True)
    fig = px.imshow(corr_df, text_auto=".2f", title="Correlation Matrix", color_continuous_scale=px.colors.sequential.Teal, aspect="auto")
    fig.update_layout(height=max(600, len(corr_df.columns) * 25), template=custom_template)
    return fig

@app.callback(
    Output("eda-hist-graph", "figure"),
    Input("eda-hist-x", "value"),
    Input("eda-hist-bins", "value")
)
def update_eda_histogram(x_ax, n_bins):
    if not x_ax:
        return go.Figure().update_layout(title_text="Please select a variable for the X-Axis", template=custom_template)
    fig = px.histogram(DF_EDA, x=x_ax, nbins=n_bins, title=f"Distribution of {x_ax}", color_discrete_sequence=['#64ffda'])
    fig.update_layout(template=custom_template)
    return fig

@app.callback(
    Output("eda-scatter-graph", "figure"),
    Input("eda-scatter-x", "value"),
    Input("eda-scatter-y", "value"),
    Input("eda-scatter-color", "value")
)
def update_eda_scatter(x_ax, y_ax, color_ax):
    if not x_ax or not y_ax:
        return go.Figure().update_layout(title_text="Please select variables for the X and Y Axes", template=custom_template)
    fig = px.scatter(DF_EDA.sample(min(len(DF_EDA), 2000)), x=x_ax, y=y_ax, color=color_ax, title=f"{y_ax} vs. {x_ax}" + (f" by {color_ax}" if color_ax else ""), opacity=0.7)
    fig.update_layout(template=custom_template)
    return fig

@app.callback(
    Output('pipeline-display-card-layman', 'children'),
    Output('pipeline-progress-layman', 'value'),
    Output('pipeline-step-store-layman', 'data'),
    Input('prev-step-button-layman', 'n_clicks'),
    Input('next-step-button-layman', 'n_clicks'),
    State('pipeline-step-store-layman', 'data')
)
def update_pipeline_step_layman(prev_clicks, next_clicks, current_step):
    button_id = ctx.triggered_id
    if button_id == 'next-step-button-layman' and current_step < len(PIPELINE_STEPS_LAYMAN) - 1:
        current_step += 1
    elif button_id == 'prev-step-button-layman' and current_step > 0:
        current_step -= 1
    return create_pipeline_step_card(PIPELINE_STEPS_LAYMAN[current_step]), current_step + 1, current_step

@app.callback(
    Output('pipeline-display-card-technical', 'children'),
    Output('pipeline-progress-technical', 'value'),
    Output('pipeline-step-store-technical', 'data'),
    Input('prev-step-button-technical', 'n_clicks'),
    Input('next-step-button-technical', 'n_clicks'),
    State('pipeline-step-store-technical', 'data')
)
def update_pipeline_step_technical(prev_clicks, next_clicks, current_step):
    button_id = ctx.triggered_id
    if button_id == 'next-step-button-technical' and current_step < len(PIPELINE_STEPS_TECHNICAL) - 1:
        current_step += 1
    elif button_id == 'prev-step-button-technical' and current_step > 0:
        current_step -= 1
    return create_pipeline_step_card(PIPELINE_STEPS_TECHNICAL[current_step]), current_step + 1, current_step

@app.callback(
    Output('pie-chart', 'figure'),
    Input({'type': 'percentage-slider', 'index': ALL}, 'value')
)
def update_pie_chart(percentages):
    if not any(v is not None for v in percentages):
        return go.Figure().update_layout(title_text="Loading...", template=custom_template)
    percentages_clean = [p or 0 for p in percentages]
    active_comps = [name for i, name in enumerate(COMPONENTS) if percentages_clean[i] > 0]
    active_percs = [p for p in percentages_clean if p > 0]
    fig = go.Figure()
    if not active_percs:
        fig.update_layout(title_text="Set component percentages", title_x=0.5)
    else:
        fig.add_trace(go.Pie(
            labels=active_comps, values=active_percs, title="Concentration", hole=0.4,
            marker=dict(colors=px.colors.qualitative.Plotly), textinfo='percent+label',
            textfont=dict(color="#0b0f1a", size=14), insidetextorientation='radial'
        ))
        fig.update_layout(showlegend=False)
    fig.update_layout(template=custom_template)
    return fig

@app.callback(
    Output('prediction-output-card', 'children'),
    Input('predict-button', 'n_clicks'),
    State({'type': 'percentage-slider', 'index': ALL}, 'value'),
    State({'type': 'prop-input', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def run_manual_prediction(n_clicks, percentages, props):
    try:
        total_pct = sum(p or 0 for p in percentages)
        if total_pct == 0:
            return dbc.Alert("Total concentration is 0%.", color="danger")
        props_array = np.array([p or 0 for p in props]).reshape(len(COMPONENTS), len(PROPERTIES))
        weights = np.array([p or 0 for p in percentages]) / total_pct
        feature_dict = {}
        for j in range(len(COMPONENTS)):
            feature_dict[f"Component{j+1}_fraction"] = weights[j]
            for i in range(len(PROPERTIES)):
                feature_dict[f"Component{j+1}_Property{i+1}"] = props_array[j, i]
                feature_dict[f"Component{j+1}_Weighted_Property{i+1}"] = weights[j] * props_array[j, i]
        for i in range(len(PROPERTIES)):
            prop_i_values = props_array[:, i]
            feature_dict[f"Weighted_Property_{i+1}"] = np.dot(weights, prop_i_values)
            feature_dict[f"Prop{i+1}_mean"] = np.mean(prop_i_values)
            feature_dict[f"Prop{i+1}_std"] = np.std(prop_i_values)
        input_df = pd.DataFrame([feature_dict])
        predictions, std_devs, base_preds_rows = {}, {}, []
        for prop_folder, ens in ENSEMBLES.items():
            try:
                final, std, stacked = ens.predict(input_df)
                predictions[prop_folder] = np.round(final.ravel()[0], 4)
                std_devs[prop_folder] = np.round(std.ravel()[0], 4)
                model_names = ["ANN", "XGB", "LGB", "CAT", "SVR", "LIN"]
                base_results = {k: v for k, v in zip(model_names, np.round(stacked.ravel(), 4))}
                base_results["Property"] = prop_folder
                base_preds_rows.append(base_results)
            except Exception as e:
                predictions[prop_folder], std_devs[prop_folder] = "Error", "Error"
                base_preds_rows.append({"Property": prop_folder, "Error": str(e)})
        pred_df = pd.DataFrame([
            {"Blend Property": p, "Predicted Value": v, "Uncertainty (Std Dev)": s}
            for p, v, s in zip(predictions.keys(), predictions.values(), std_devs.values())
        ])
        base_df = pd.DataFrame(base_preds_rows)
        return [
            dbc.CardHeader(html.H4("✅ Prediction Results")),
            dbc.CardBody([
                dbc.Label("Final Predictions with Uncertainty"),
                dash_table.DataTable(data=pred_df.to_dict('records'), columns=[{'name': i, 'id': i} for i in pred_df.columns], **datatable_style),
                html.Hr(),
                dbc.Button("Show/Hide Base Learner Breakdown", id="collapse-button", className="mb-3", color="secondary", outline=True),
                dbc.Collapse(
                    dash_table.DataTable(data=base_df.to_dict('records'), columns=[{'name': i, 'id': i} for i in base_df.columns], **datatable_style),
                    id="collapse-base-learners",
                    is_open=False,
                ),
            ], className="fade-in")
        ]
    except Exception as e:
        return dbc.Alert(f"An unexpected error occurred: {e}", color="danger")

@app.callback(
    Output("collapse-base-learners", "is_open"),
    Input("collapse-button", "n_clicks"),
    State("collapse-base-learners", "is_open"),
    prevent_initial_call=True,
)
def toggle_collapse(n, is_open):
    return not is_open if n else is_open
    
@app.callback(
    Output('model-details-output', 'children'),
    Input('model-details-dropdown', 'value')
)
def update_model_details(property_folder):
    if not ENSEMBLES or property_folder not in ENSEMBLES:
        return dbc.Alert("No model loaded for this property.", color="warning")
    ens = ENSEMBLES[property_folder]
    ann_net = ens.ann.net
    ann_layers = html.Ul([
        html.Li(f"Input Layer: {ann_net[0].in_features} features"),
        html.Li(f"Hidden Layer 1 (ReLU): {ann_net[0].out_features} neurons"),
        html.Li(f"Hidden Layer 2 (ReLU): {ann_net[4].out_features} neurons"),
        html.Li(f"Output Layer: 1 neuron (Regression)")
    ])
    ensemble_items = []
    for model_name, model in ens.models.items():
        params_dict = {k: v for k, v in model.get_params().items() if v is not None}
        params_df = pd.DataFrame(list(params_dict.items()), columns=['Parameter', 'Value'])
        ensemble_items.append(dbc.Card(dbc.CardBody([
            html.H5(f"Base Learner: {model_name.upper()}"),
            dash_table.DataTable(data=params_df.to_dict('records'), columns=[{"name": i, "id": i} for i in params_df.columns], **datatable_style)
        ]), className="mt-2"))
    meta_model_info = {k: v for k, v in ens.meta.get_params().items() if v is not None}
    meta_df = pd.DataFrame(list(meta_model_info.items()), columns=['Parameter', 'Value'])
    ensemble_items.append(dbc.Card(dbc.CardBody([
        html.H5("Meta-Model (Final Stacker)"),
        dash_table.DataTable(data=meta_df.to_dict('records'), columns=[{"name": i, "id": i} for i in meta_df.columns], **datatable_style)
    ]), className="mt-2"))
    return html.Div([
        html.H4("Artificial Neural Network (ANN) Architecture"),
        html.P("A multi-layer perceptron architecture for regression."),
        ann_layers,
        html.Hr(),
        html.H4("Ensemble Stack Architecture"),
        html.P("A meta-model combines predictions from the following base learners:"),
        html.Div(ensemble_items)
    ])
    
# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    app.run(debug=True, port=8050)