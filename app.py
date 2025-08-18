# app.py
# FINAL AESTHETIC VERSION - (Accepts Headerless CSV)

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
APP_TITLE = "Fuel Blend Ensemble Predictor"

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
    def __init__(self, folder: str):
        self.folder = folder
        # Scalers
        self.x_scaler = joblib.load(os.path.join(folder, "x_scaler.pkl"))
        self.y_scaler = joblib.load(os.path.join(folder, "y_scaler.pkl"))
        # Classical models
        models_to_load = [("xgb", "xgb.pkl"), ("lgb", "lgb.pkl"), ("svr", "svr.pkl"), ("lin", "lin.pkl")]
        self.models = {name: joblib.load(os.path.join(folder, fname)) for name, fname in models_to_load}
        # CatBoost (load_model returns None)
        cat_model = CatBoostRegressor()
        cat_model.load_model(os.path.join(folder, "cat.cbm"))
        self.models["cat"] = cat_model
        # Meta model and ANN
        self.meta = joblib.load(os.path.join(folder, "meta.pkl"))
        self.ann = load_ann_model(os.path.join(folder, "ann.pt"))
        self.input_cols = list(getattr(self.x_scaler, "feature_names_in_", []))

    def _align_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.input_cols:
            return df
        df_aligned = df.copy()
        for col in self.input_cols:
            if col not in df_aligned.columns:
                df_aligned[col] = 0
        return df_aligned[self.input_cols]

    def predict(self, df: pd.DataFrame):
        X_aligned = self._align_df(df)
        X_scaled = self.x_scaler.transform(X_aligned.values)
        # ANN expects scaled inputs
        with torch.no_grad():
            ann_out_scaled = self.ann(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
        base_preds_scaled = [ann_out_scaled]
        for name in ["xgb", "lgb", "cat", "svr", "lin"]:
            raw = self.models[name].predict(X_scaled)
            base_preds_scaled.append(raw.reshape(-1, 1) if getattr(raw, 'ndim', 1) == 1 else raw)
        # Inverse-scale each base model output independently
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
    suppress_callback_exceptions=True
)
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

# Light theme variant for Plotly
custom_template_light = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1b2a41"),
        title_font=dict(color="#1b2a41"),
        legend_font=dict(color="#1b2a41"),
        xaxis={"gridcolor": "rgba(27, 42, 65, 0.08)", "linecolor": "rgba(27, 42, 65, 0.15)"},
        yaxis={"gridcolor": "rgba(27, 42, 65, 0.08)", "linecolor": "rgba(27, 42, 65, 0.15)"}
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
    {"title": "Step 1 & 2: Raw Data & Feature Engineering", "text": "The system starts with a ⁠ 5x10 ⁠ matrix of component properties and a ⁠ 1x5 ⁠ vector of component concentrations. This raw input is transformed into a richer feature set. The pipeline calculates a wide array of new features, including: weighted averages for each property, and statistical features such as the ⁠ mean ⁠ and ⁠ standard deviation ⁠ of all 10 properties across the 5 components. This enriched feature set is then standardized using a ⁠ StandardScaler ⁠.", "image": "tech_step1_feature_engineering.png", "icon": "bi bi-keyboard-fill"},
    {"title": "Step 3: Base Learner Predictions (Level-0 Models)", "text": "The standardized features are fed into a diverse collection of base models. These models, though trained for the same task, use different algorithms and, therefore, learn different patterns from the data. This diversity is key to ensemble learning. The base learners in this ensemble are a *Neural Network (ANN), **XGBoost, **LightGBM, **CatBoost, **SVR, and **Linear Regression*. Each of these models outputs its own prediction for the target property.", "image": "tech_step2_base_learners.png", "icon": "bi bi-people-fill"},
    {"title": "Step 4: Stacking & Meta-Model (Level-1 Model)", "text": "This is the core of the ensemble. The predictions from the six base learners are not directly averaged. Instead, they are 'stacked' together to form a new input vector. This new input vector is given to a final *meta-model* which is trained to find the optimal way to combine the base predictions, correcting their biases and reducing their variance.", "image": "tech_step3_stacking_meta_model.png", "icon": "bi bi-person-check-fill"},
    {"title": "Step 5: Final Ensemble Prediction", "text": "The output from the meta-model is the final, definitive prediction for the fuel blend property. This stacked ensemble approach is a form of *stacked generalization*, which is widely recognized as one of the most powerful techniques in machine learning for improving predictive performance.", "image": "tech_step4_final_prediction.png", "icon": "bi bi-trophy-fill"},
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
                dbc.Col(dcc.Slider(
                    min=0, max=100, step=1, value=20 if comp_idx == 0 else 0,
                    updatemode='mouseup',  # update on release for smoother UI
                    marks={0: '0', 25: '25', 50: '50', 75: '75', 100: '100'},
                    id={'type': 'percentage-slider', 'index': comp_idx}
                ), md=9, xs=8),
                dbc.Col(dbc.InputGroup([
                    dbc.Input(
                        type="number", min=0, max=100, step=1,
                        value=20 if comp_idx == 0 else 0,
                        debounce=True,  # reduce rapid callbacks while typing
                        id={'type': 'percentage-input', 'index': comp_idx}
                    ),
                ], className="mt-1 mt-md-0"), md=3, xs=4)
            ], align="center"),
            html.Hr(),
            dbc.Row([
                dbc.Col(dbc.InputGroup([
                    dbc.InputGroupText(f"P{prop_idx+1}"),
                    dbc.Input(
                        type="number", value=0, step=0.01,
                        id={'type': 'prop-input', 'index': f"{comp_idx}-{prop_idx}"}
                    )
                ]), md=6) for prop_idx in range(len(PROPERTIES))
            ], className="g-2")
        ]
    )

def parse_contents_headerless(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Read the CSV with NO header
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
        return dbc.Alert("This is not a CSV file!", color="danger")
    except Exception as e:
        return dbc.Alert(f'There was an error processing this file: {e}', color="danger")

# ========== APP LAYOUT ==========
tab_style = {
    'border': 'none',
    'border-bottom': '2px solid transparent', 'border-radius': '12px 12px 0 0',
    'padding': '12px 20px', 'marginRight': '4px',
}
active_tab_style = {
    'fontWeight': 'bold'
}
def get_datatable_style(theme: str = "dark"):
    is_light = (theme == "light")
    base = {
        'style_as_list_view': True,
        'style_cell': {
            'textAlign': 'left', 
            'padding': '12px', 
            'backgroundColor': '#ffffff' if is_light else '#1a324a',
            'color': '#1b2a41' if is_light else '#f0f8ff', 
            'border': '1px solid ' + ('#dbe7f3' if is_light else '#254a6b'),
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '120px',
            'width': 'auto',
            'maxWidth': 'none',
            'fontSize': '14px',
            'lineHeight': '1.4'
        },
        'style_header': {
            'backgroundColor': '#e9f4ff' if is_light else '#254a6b', 
            'color': '#0d1a26' if is_light else 'white', 
            'fontWeight': 'bold',
            'textAlign': 'center',
            'padding': '12px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'fontSize': '14px'
        },
        'style_data': {
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '1.4'
        },
    }
    return base

datatable_style = get_datatable_style("dark")
upload_style = {
    'width': '100%', 
    'height': 'auto', 
    'minHeight': '100px',
    'borderWidth': '2px',
    'borderStyle': 'dashed', 
    'borderRadius': '15px', 
    'textAlign': 'center',
    'margin-bottom': '20px', 
    'borderColor': 'rgba(100, 255, 218, 0.4)',
    'backgroundColor': 'rgba(30, 70, 110, 0.2)', 
    'color': '#e0f2fe',
    'fontSize': '0.9rem', 
    'fontWeight': '500', 
    'cursor': 'pointer',
    'transition': 'all 0.3s ease',
    'padding': '20px',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'center'
}

app.layout = html.Div(id="theme-root", className="theme-dark perf-smooth", children=[
    dcc.Store(id='theme-store', data='dark'),
    dcc.Store(id='pipeline-step-store-layman', data=0),
    dcc.Store(id='pipeline-step-store-technical', data=0),
    dcc.Store(id='batch-pred-store'),
    # Removed auto-redirect so the app lands and stays on Overview

    # Enhanced Navbar with better branding
    dbc.NavbarSimple(
        brand=html.Div([
            html.I(className="bi bi-rocket-takeoff me-2"),
            APP_TITLE
        ]), 
        color="primary", 
        dark=True, 
        className="mb-4",
        style={'position': 'sticky', 'top': '0', 'z-index': '1000'}
    ),

    # Floating Theme Toggle
    html.Div(
        html.Div([
            html.I(className="bi bi-brightness-high me-2"),
            dbc.Checklist(
                id="theme-toggle",
                options=[{"label": "Light", "value": 1}],
                value=[],
                switch=True,
                className="theme-toggle-switch"
            )
        ], className="theme-toggle-fab")
    ),

    # Main content container (padded)
    dbc.Container([
        # Tabs with Overview + sections
        dbc.Tabs(id="main-tabs", active_tab="tab-home", className="mb-4 dash-tabs", children=[
            dbc.Tab(
                label="🏠 Overview", 
                tab_id="tab-home",
                tab_style=tab_style, active_tab_style=active_tab_style,
                children=html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H1("EcoFuelFusion.ai", className="section-title mb-3"),
                                    html.P("Accurate, explainable fuel blend predictions powered by a stacked ensemble of ML models.", className="lead mb-4"),
                                    html.Ul([
                                        html.Li("Blend property predictions across 10 targets"),
                                        html.Li("Transparent model details and parameter explorer"),
                                        html.Li("Interactive EDA tools for fast insight")
                                    ], className="mb-4"),
                                    dbc.Button("Get Started →", id="get-started-btn", color="primary", size="lg")
                                ])
                            ], className="fade-in hero-card")
                        ], md=7),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id="overview-graphic", figure=go.Figure())
                                ])
                            ], className="fade-in float")
                        ], md=5)
                    ], className="g-4 align-items-stretch")
                ], className="mt-3")
            ),
            dbc.Tab(
                label="🔧 Prediction Workbench", 
                tab_id="tab-workbench",
                tab_style=tab_style, active_tab_style=active_tab_style,
                children=dbc.Row([
                    dbc.Col([
                        dbc.Card(className="fade-in slide-in-left", children=[
                            dbc.CardHeader([
                                html.H4(html.Div([
                                    html.I(className="bi bi-flask me-2"),
                                    "🧪 Input Workbench"
                                ]), className="mb-0")
                            ]),
                            dbc.CardBody([
                                # Enhanced upload area
                                html.Div([
                                    html.H6("📁 Data Import", className="mb-3", style={'color': '#64ffda'}),
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            html.Div([
                                                html.I(className="bi bi-cloud-upload fs-2 mb-2", style={'display': 'block'}),
                                                html.Div([
                                                    'Drag and Drop a CSV file or ',
                                                    html.A('Browse Files', style={'color': '#64ffda', 'textDecoration': 'underline'})
                                                ], style={'fontSize': '0.9rem'})
                                            ], style={'textAlign': 'center'})
                                        ], style={'height': '100%', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                        style=upload_style,
                                        multiple=False,
                                        className="upload-area"
                                    ),
                                    html.Div(id='upload-message', className='mb-4'),
                                    # Batch predictions upload (multi-row, headerless)
                                    html.Hr(style={'borderColor': 'rgba(135, 206, 235, 0.2)'}),
                                    html.H6("📦 Batch Predictions (Headerless CSV)", className="mb-2", style={'color': '#64ffda'}),
                                    dcc.Upload(
                                        id='upload-batch',
                                        children=html.Div([
                                            html.Div([
                                                html.I(className="bi bi-upload fs-5 me-2"),
                                                html.Span("Upload a CSV with N rows × 55+ columns (5 concentrations + 50 properties)")
                                            ])
                                        ]),
                                        style={**upload_style, 'minHeight': '70px', 'padding': '12px'},
                                        multiple=False,
                                        className="upload-area"
                                    ),
                                ]),
                                
                                html.Hr(style={'borderColor': 'rgba(135, 206, 235, 0.3)', 'margin': '30px 0'}),
                                
                                # Manual input section
                                html.Div([
                                    html.H6("⚙️ Manual Configuration", className="mb-3", style={'color': '#64ffda'}),
                                    dbc.Accordion(
                                        [make_input_card(i, name) for i, name in enumerate(COMPONENTS)],
                                        always_open=True, active_item="item-0"
                                    ),
                                ]),
                                
                                # Enhanced prediction button
                                html.Div([
                                    dbc.Button(
                                        html.Div([
                                            html.I(className="bi bi-cpu me-2"), 
                                            "🚀 Run AI Predictions"
                                        ]),
                                        id='predict-button', 
                                        n_clicks=0, 
                                        className="w-100 mt-4",
                                        size="lg", 
                                        color="primary",
                                        style={'fontSize': '1.1rem', 'padding': '15px'}
                                    )
                                ], className="text-center")
                            ])
                        ])
                    ], md=6),
                    
                    dbc.Col([
                                dcc.Loading(
                            children=dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H5(html.Div([
                                                html.I(className="bi bi-pie-chart me-2"),
                                                "� Composition Overview"
                                            ]), className="mb-0")
                                        ]),
                                        dbc.CardBody([
                                            dcc.Graph(id='pie-chart')
                                        ])
                                    ], className="fade-in slide-in-right float")
                                ], width=12, className="mb-4"),
                                
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H5(html.Div([
                                                html.I(className="bi bi-graph-up me-2"),
                                                "🎯 Prediction Results"
                                            ]), className="mb-0")
                                        ]),
                                        dbc.CardBody(
                                            id="prediction-output-card",
                                            children=[
                                                html.Div([
                                                    html.I(className="bi bi-info-circle fs-1 mb-3", style={'color': '#64ffda'}),
                                                    html.H5("Ready for Predictions", className="mb-3"),
                                                    html.P("Configure your fuel blend components and click 'Run AI Predictions' to see detailed results.", 
                                                          className="text-muted")
                                                ], className="text-center py-4")
                                            ],
                                            className="fade-in"
                                        )
                                    ], className="fade-in slide-in-right")
                                ], width=12),

                                # Batch predictions results card (right column)
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H5(html.Div([
                                                html.I(className="bi bi-file-spreadsheet me-2"),
                                                "📦 Batch Predictions"
                                            ]), className="mb-0")
                                        ]),
                                        dbc.CardBody(id='batch-pred-result', children=[
                                            html.Div([
                                                html.I(className="bi bi-info-circle fs-1 mb-3", style={'color': '#64ffda'}),
                                                html.H5("Upload a CSV for Batch Predictions", className="mb-3"),
                                                html.P("Use the second upload above to process multiple blends at once. We'll show a summary table here and let you download output.csv.", className='text-muted')
                                            ], className='text-center py-4'),
                                            dcc.Download(id='download-batch-csv')
                                        ], className="fade-in"),
                                    ], className="fade-in slide-in-right")
                                ], width=12)
                            ]),
                            type="default",
                            style={'minHeight': '100px'}
                        )
                    ], md=6),
                ], className="mt-4")
            ),
            dbc.Tab(
                label="🔬 Model Details", 
                tab_id="tab-model-details",
                tab_style=tab_style, active_tab_style=active_tab_style,
                children=html.Div([
                    html.Div([
                        html.H2("🔬 Ensemble Architecture Deep Dive", className="section-title"),
                        html.P("Explore the sophisticated machine learning models powering our predictions.", 
                               className="lead text-center mb-4")
                    ], className="modern-container fade-in"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H4(html.Div([
                                        html.I(className="bi bi-diagram-3 me-2"),
                                        "Model Architecture Explorer"
                                    ]), className="card-title mb-0")
                                ]),
                                dbc.CardBody([
                                    html.P("Select a blend property to examine its underlying ensemble architecture and model parameters.", 
                                          className="mb-4"),
                                    dbc.Label("🎯 Target Property", className="fw-bold mb-2"),
                                    dcc.Dropdown(
                                        id="model-details-dropdown",
                                        options=[{'label': k, 'value': k} for k in ENSEMBLES.keys()],
                                        value=list(ENSEMBLES.keys())[0] if ENSEMBLES else None,
                                        clearable=False, 
                                        className="mb-4",
                                        style={'fontSize': '1rem'}
                                    ),
                                    html.Hr(),
                                    dcc.Loading(
                                        id="model-details-output",
                                        type="default",
                                        style={'minHeight': '200px'}
                                    )
                                ])
                            ], className="fade-in")
                        ], width=12)
                    ], className="mt-4")
                ])
            ),
            dbc.Tab(
                label="💡 How it Works", 
                tab_id="tab-layman-intuition",
                tab_style=tab_style, active_tab_style=active_tab_style,
                children=html.Div([
                    html.Div([
                        html.H2("🧠 The Prediction Pipeline: A Simple Guide", className="section-title"),
                        html.P("Understanding how AI makes fuel blend predictions, explained in simple terms.", 
                               className="lead text-center mb-4")
                    ], className="modern-container fade-in"),
                    
                    dbc.Card(
                        create_pipeline_step_card(PIPELINE_STEPS_LAYMAN[0]), 
                        id="pipeline-display-card-layman", 
                        body=True,
                        className="fade-in"
                    ),
                    dbc.Progress(
                        id="pipeline-progress-layman", 
                        value=1, 
                        max=len(PIPELINE_STEPS_LAYMAN), 
                        className="my-4",
                        style={'height': '8px'}
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(html.Div([
                                html.I(className="bi bi-arrow-left me-2"), 
                                "Previous"
                            ]), id="prev-step-button-layman", outline=True, color="info", className="w-100")
                        ]),
                        dbc.Col([
                            dbc.Button(html.Div([
                                "Next", 
                                html.I(className="bi bi-arrow-right ms-2")
                            ]), id="next-step-button-layman", outline=True, color="info", className="w-100")
                        ]),
                    ], justify="between", className="pipeline-nav")
                ], className="mt-4")
            ),
            dbc.Tab(
                label="⚙️ Technical Details", 
                tab_id="tab-technical-overview",
                tab_style=tab_style, active_tab_style=active_tab_style,
                children=html.Div([
                    html.Div([
                        html.H2("⚙️ The Prediction Pipeline: Technical Details", className="section-title"),
                        html.P("Deep dive into the machine learning architecture and ensemble methodology.", 
                               className="lead text-center mb-4")
                    ], className="modern-container fade-in"),
                    
                    dbc.Card(
                        create_pipeline_step_card(PIPELINE_STEPS_TECHNICAL[0]), 
                        id="pipeline-display-card-technical", 
                        body=True,
                        className="fade-in"
                    ),
                    dbc.Progress(
                        id="pipeline-progress-technical", 
                        value=1, 
                        max=len(PIPELINE_STEPS_TECHNICAL), 
                        className="my-4",
                        style={'height': '8px'}
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(html.Div([
                                html.I(className="bi bi-arrow-left me-2"), 
                                "Previous"
                            ]), id="prev-step-button-technical", outline=True, color="info", className="w-100")
                        ]),
                        dbc.Col([
                            dbc.Button(html.Div([
                                "Next", 
                                html.I(className="bi bi-arrow-right ms-2")
                            ]), id="next-step-button-technical", outline=True, color="info", className="w-100")
                        ]),
                    ], justify="between", className="pipeline-nav")
                ], className="mt-4")
            ),
            dbc.Tab(
                label="📊 Data Analytics", 
                tab_id="tab-eda",
                tab_style=tab_style, active_tab_style=active_tab_style,
                children=html.Div([
                    html.Div([
                        html.H2("📈 Exploratory Data Analysis", className="section-title"),
                        html.P("Interactive data visualization and statistical analysis of fuel blend properties.", 
                               className="lead text-center mb-4")
                    ], className="modern-container fade-in"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Tabs(
                                id="eda-sub-tabs",
                                active_tab="eda-tab-corr",
                                className="dash-tabs",
                                children=[
                                    dbc.Tab(label="🔥 Correlation Heatmap", tab_id="eda-tab-corr", tab_style=tab_style, active_tab_style=active_tab_style, children=[
                                        dcc.Loading(
                                            dcc.Graph(id="eda-graph-corr", style={"height": "80vh"}),
                                            type="default"
                                        )
                                    ]),
                                    dbc.Tab(label="📊 Distribution Histogram", tab_id="eda-tab-hist", tab_style=tab_style, active_tab_style=active_tab_style, children=[
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody([
                                                        dbc.Label("Variable", className="fw-bold mb-2"),
                                                        dcc.Dropdown(
                                                            id="eda-hist-x",
                                                            options=[
                                                                {"label": c, "value": c} for c in sorted([
                                                                    col for col in DF_EDA.columns
                                                                    if ("Component" in col or "BlendProperty" in col) and "Weighted" not in col
                                                                ])
                                                            ] if not DF_EDA.empty else [],
                                                            value=None,
                                                            optionHeight=36,
                                                            className="mb-3 eda-hist-dropdown"
                                                        )
                                                    ]),
                                                    className="mb-3 eda-card"
                                                ),
                                                width=12
                                            )
                                        ]),
                                        dcc.Loading(
                                            dcc.Graph(id="eda-graph-hist", style={"height": "80vh"}),
                                            type="default"
                                        )
                                    ]),
                    dbc.Tab(label="🔵 Scatter Plots", tab_id="eda-tab-scatter", tab_style=tab_style, active_tab_style=active_tab_style, children=[
                                        dbc.Row([
                                            dbc.Col(
                        dbc.Card(
                                                    dbc.CardBody([
                                                        dbc.Label("X-Axis Variable", className="fw-bold mb-2"),
                                                        dcc.Dropdown(
                                                            id="eda-scatter-x",
                                                            options=[
                                                                {"label": c, "value": c} for c in sorted(DF_EDA.columns)
                                                            ] if not DF_EDA.empty else [],
                                value=None,
                                optionHeight=36,
                                className="mb-3 eda-scatter-dropdown"
                                                        ),
                                                        dbc.Label("Y-Axis Variable", className="fw-bold mb-2 mt-3"),
                                                        dcc.Dropdown(
                                                            id="eda-scatter-y",
                                                            options=[
                                                                {"label": c, "value": c} for c in sorted(DF_EDA.columns)
                                                            ] if not DF_EDA.empty else [],
                                value=None,
                                optionHeight=36,
                                className="mb-3 eda-scatter-dropdown"
                                                        ),
                                                    ]),
                            className="mb-3 eda-card"
                                                ),
                                                width=12
                                            )
                                        ]),
                                        dcc.Loading(
                                            dcc.Graph(id="eda-graph-scatter", style={"height": "80vh"}),
                                            type="default"
                                        )
                                    ]),
                                ]
                            )
                        ])
                    ], body=True, className="mt-4 fade-in")
                ])
            ),
        ])
    ], fluid=True),

    # Full-width Copyright Footer (outside container to avoid side padding)
    html.Footer([
        html.Div([
            html.Div([
                html.Div([
                    html.Span("© 2025 EcoFuelFusion.ai | All rights reserved.", className="mb-0"),
                    html.Span(" | Built with ❤️ using Dash & Python", className="app-version")
                ]),
                html.Div([
                    html.A("Documentation", href="https://github.com/ThePrachiShuk/EcoFuelFusion/blob/main/README.md", className="me-3"),
                    html.A("GitHub", href="https://github.com/ThePrachiShuk/EcoFuelFusion", className="me-3"),
                    html.A("Support", href="#"),
                ], className="footer-links")
            ], className="footer-content")
        ])
    ], className="copyright-footer")
])

# ========== CALLBACKS ==========

# Unified sync: mirrors slider<->input changes and handles CSV upload population
@app.callback(
    Output({'type': 'percentage-slider', 'index': ALL}, 'value'),
    Output({'type': 'percentage-input', 'index': ALL}, 'value'),
    Output({'type': 'prop-input', 'index': ALL}, 'value'),
    Output('upload-message', 'children'),
    Input({'type': 'percentage-slider', 'index': ALL}, 'value'),
    Input({'type': 'percentage-input', 'index': ALL}, 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State({'type': 'prop-input', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def sync_percentages_and_upload(slider_vals, input_vals, contents, filename, prop_vals_state):
    trig = ctx.triggered_id
    slider_n = len(slider_vals or [])
    input_n = len(input_vals or [])
    prop_n = len(prop_vals_state or [])
    # Helpers: list-shaped no_update for ALL-pattern outputs
    no_sliders = [no_update] * slider_n
    no_inputs = [no_update] * input_n
    no_props = [no_update] * prop_n
    # Upload handler
    if trig == 'upload-data' and contents is not None:
        df = parse_contents_headerless(contents, filename)
        if not isinstance(df, pd.DataFrame):
            return no_sliders, no_inputs, no_props, df
        if len(df) != 1 or len(df.columns) < 55:
            error_msg = dbc.Alert("Error: CSV must have one row with at least 55 columns.", color="danger")
            return no_sliders, no_inputs, no_props, error_msg
        try:
            row = df.iloc[0]
            sliders_raw = [max(0, min(100, float(row[i]))) for i in range(5)]
            sliders = [int(round(v)) for v in sliders_raw[:slider_n]]
            sliders_inputs = [int(round(v)) for v in sliders_raw[:input_n]]
            props_raw = [float(row[i]) for i in range(5, 55)]
            props = props_raw[:prop_n]
            msg = dbc.Alert(f"Successfully loaded headerless data from {filename}", color="success")
            return sliders, sliders_inputs, props, msg
        except Exception as e:
            error_msg = dbc.Alert(f"Error processing data by position: {e}", color="danger")
            return no_sliders, no_inputs, no_props, error_msg

    # Slider changed -> mirror to inputs only
    if isinstance(trig, dict) and trig.get('type') == 'percentage-slider':
        # Target inputs cast to ints
        i_target = [int(v or 0) for v in (slider_vals or [])][:input_n]
        i_current = [
            (int(v) if v not in (None, "") else 0) for v in (input_vals or [])
        ][:input_n]
        if i_current == i_target:
            return no_sliders, no_inputs, no_props, no_update
        return no_sliders, i_target, no_props, no_update

    # Number input changed -> sanitize and mirror to sliders only
    if isinstance(trig, dict) and trig.get('type') == 'percentage-input':
        sanitized = [
            max(0, min(100, int(float(v)) if v not in (None, "") else 0))
            for v in (input_vals or [])
        ]
        s_target = sanitized[:slider_n]
        s_current = [int(v or 0) for v in (slider_vals or [])][:slider_n]
        if s_current == s_target:
            return no_sliders, no_inputs, no_props, no_update
        return s_target, no_inputs, no_props, no_update
    
    return no_sliders, no_inputs, no_props, no_update


@app.callback(
    Output("eda-graph-corr", "figure"),
    Output("eda-graph-hist", "figure"),
    Output("eda-graph-scatter", "figure"),
    Input("eda-sub-tabs", "active_tab"),
    Input("eda-hist-x", "value"),
    Input("eda-scatter-x", "value"),
    Input("eda-scatter-y", "value"),
    Input('theme-store', 'data')
)
def update_eda_graphs(active_tab, hist_x, scatter_x, scatter_y, theme):
    tpl = custom_template_light if theme == 'light' else custom_template
    empty_fig = go.Figure().update_layout(template=tpl)
    if DF_EDA.empty:
        nf = go.Figure().update_layout(title_text="Dataset Not Loaded", template=tpl)
        return nf, nf, nf

    # Correlation
    corr = DF_EDA.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr, text_auto=".2f", title="Correlation Matrix",
        color_continuous_scale=px.colors.sequential.Teal
    )
    fig_corr.update_layout(height=max(600, len(corr.columns) * 25), template=tpl)

    # Histogram (theme-consistent colors + subtle outline)
    if hist_x:
        bar_color = '#90caf9' if theme == 'light' else '#64ffda'
        line_color = 'rgba(27,42,65,0.55)' if theme == 'light' else 'rgba(255,255,255,0.28)'
        fig_hist = px.histogram(
            DF_EDA, x=hist_x, title=f"Distribution of {hist_x}", nbins=30,
            color_discrete_sequence=[bar_color]
        )
        fig_hist.update_traces(marker=dict(color=bar_color, opacity=0.9,
                               line=dict(color=line_color, width=0.8)))
        fig_hist.update_layout(template=tpl, bargap=0.05)
    else:
        fig_hist = empty_fig.update_layout(title_text="Select a variable for histogram")

    # Scatter
    if scatter_x and scatter_y:
        # Theme-aware point styling for visibility and consistency
        point_color = '#1976d2' if theme == 'light' else '#64ffda'  # blue in light, teal in dark
        outline_color = 'rgba(27,42,65,0.6)' if theme == 'light' else 'rgba(255,255,255,0.35)'
        fig_scatter = px.scatter(
            DF_EDA, x=scatter_x, y=scatter_y,
            title=f"Scatter: {scatter_x} vs {scatter_y}",
            color_discrete_sequence=[point_color]
        )
        fig_scatter.update_traces(
            marker=dict(size=7, opacity=0.85, line=dict(color=outline_color, width=0.8))
        )
        fig_scatter.update_layout(template=tpl)
    else:
        fig_scatter = empty_fig.update_layout(title_text="Select X and Y for scatter")

    return fig_corr, fig_hist, fig_scatter


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
    Input({'type': 'percentage-slider', 'index': ALL}, 'value'),
    Input('theme-store', 'data')
)
def update_pie_chart(percentages, theme):
    tpl = custom_template_light if theme == 'light' else custom_template
    if not any(v is not None for v in percentages):
        return go.Figure().update_layout(title_text="Loading...", template=tpl)
    percentages_clean = [p or 0 for p in percentages]
    active_comps = [name for i, name in enumerate(COMPONENTS) if percentages_clean[i] > 0]
    active_percs = [p for p in percentages_clean if p > 0]
    fig = go.Figure()
    if not active_percs:
        fig.update_layout(title_text="Set component percentages", title_x=0.5)
    else:
        # Theme-aware text color for labels (inside and outside)
        text_color = '#1b2a41' if theme == 'light' else '#e6f2ff'
        fig.add_trace(go.Pie(
            labels=active_comps, values=active_percs, title="Concentration", hole=0.4,
            marker=dict(colors=px.colors.qualitative.Plotly), textinfo='percent+label',
            textfont=dict(color=text_color, size=14),
            insidetextfont=dict(color=text_color),
            outsidetextfont=dict(color=text_color),
            insidetextorientation='radial'
        ))
        fig.update_layout(showlegend=False)

    fig.update_layout(template=tpl)
    return fig


@app.callback(
    Output('prediction-output-card', 'children'),
    Input('predict-button', 'n_clicks'),
    State({'type': 'percentage-slider', 'index': ALL}, 'value'),
    State({'type': 'prop-input', 'index': ALL}, 'value'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)
def run_manual_prediction(n_clicks, percentages, props, theme):
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

        table_style = get_datatable_style(theme)
        return [
            dbc.CardHeader([
                html.H4(html.Div([
                    html.I(className="bi bi-check-circle-fill me-2 text-success"),
                    "Prediction Results"
                ]), className="mb-0")
            ]),
            dbc.CardBody([
                html.Div([
                    html.H5(html.Div([
                        html.I(className="bi bi-target me-2"),
                        "🎯 Final Ensemble Predictions"
                    ]), className="mb-3"),
                    html.P("These values represent the final predictions from our ensemble of 6 ML models, with uncertainty estimates.", 
                           className="text-muted mb-4"),
                    html.Div([
                        dash_table.DataTable(
                            data=pred_df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in pred_df.columns],
                            **table_style,
                            style_table={'overflowX': 'auto', 'minWidth': '100%'},
                            page_size=10
                        )
                    ])
                ]),
                
                html.Hr(style={'margin': '30px 0', 'borderColor': 'rgba(135, 206, 235, 0.3)'}),
                
                # Collapsible Individual Model Breakdown Section
                dbc.Accordion([
                    dbc.AccordionItem(
                        title=html.Div([
                            html.I(className="bi bi-diagram-3 me-2"),
                            "🔍 Individual Model Breakdown",
                            html.Small(" (Click to expand detailed predictions)", className="ms-2 text-muted")
                        ]),
                        item_id="model-breakdown-accordion",
                        children=[
                            html.P([
                                "Detailed predictions from each base learner in the ensemble before meta-model combination. ",
                                html.Strong("→ Scroll horizontally to view all model columns →", 
                                          style={'color': '#64ffda', 'fontSize': '0.9rem'})
                            ], className="text-muted mb-4"),
                            html.Div([
                                dash_table.DataTable(
                                    data=base_df.to_dict('records'),
                                    columns=[{'name': i, 'id': i} for i in base_df.columns],
                                    **table_style,
                                    style_table={
                                        'overflowX': 'scroll', 
                                        'minWidth': '100%'
                                    },
                                    page_size=10
                                )
                            ], style={'overflowX': 'scroll', 'width': '100%'})
                        ]
                    )
                ], 
                start_collapsed=True,  # Start collapsed by default
                className="mb-4"
                )
            ], className="fade-in prediction-updated")
        ]
    except Exception as e:
        return dbc.Alert(f"An unexpected error occurred: {e}", color="danger")


# ===== Batch Predictions (multi-row CSV) =====
def _build_features_from_row(row_values: np.ndarray):
    # Expect: first 5 = concentrations, next 50 = 5 components × 10 properties
    if row_values.shape[0] < 55:
        raise ValueError("Row must have at least 55 values: 5 percentages + 50 properties")
    percentages = np.array(row_values[:5], dtype=float)
    props_flat = np.array(row_values[5:55], dtype=float)
    props_array = props_flat.reshape(len(COMPONENTS), len(PROPERTIES))
    total_pct = np.sum(np.nan_to_num(percentages))
    if total_pct <= 0:
        raise ValueError("Total concentration is 0")
    weights = percentages / total_pct
    feature_dict = {}
    for j in range(len(COMPONENTS)):
        feature_dict[f"Component{j+1}_fraction"] = float(weights[j])
        for i in range(len(PROPERTIES)):
            val = float(props_array[j, i])
            feature_dict[f"Component{j+1}_Property{i+1}"] = val
            feature_dict[f"Component{j+1}_Weighted_Property{i+1}"] = float(weights[j] * val)
    for i in range(len(PROPERTIES)):
        prop_i_values = props_array[:, i]
        feature_dict[f"Weighted_Property_{i+1}"] = float(np.dot(weights, prop_i_values))
        feature_dict[f"Prop{i+1}_mean"] = float(np.mean(prop_i_values))
        feature_dict[f"Prop{i+1}_std"] = float(np.std(prop_i_values))
    return feature_dict


@app.callback(
    Output('batch-pred-store', 'data'),
    Output('batch-pred-result', 'children'),
    Input('upload-batch', 'contents'),
    State('upload-batch', 'filename'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)
def on_batch_upload(contents, filename, theme):
    if not contents:
        raise dash.exceptions.PreventUpdate
    # Parse CSV as headerless
    df = parse_contents_headerless(contents, filename)
    if not isinstance(df, pd.DataFrame):
        # df is an Alert
        return None, df
    if df.shape[1] < 55:
        return None, dbc.Alert("CSV must have at least 55 columns (5 percentages + 50 properties)", color="danger")

    # Build features per row
    feats = []
    errors = []
    for idx, row in df.iterrows():
        try:
            feats.append(_build_features_from_row(row.values))
        except Exception as e:
            errors.append((idx, str(e)))
            feats.append(None)

    # Predict per row
    results = []
    for idx, fdict in enumerate(feats):
        if fdict is None:
            results.append({"Row": idx, "Error": errors[-1][1] if errors else "Invalid row"})
            continue
        input_df = pd.DataFrame([fdict])
        row_out = {"Row": idx}
        for prop_folder, ens in ENSEMBLES.items():
            try:
                final, std, _ = ens.predict(input_df)
                row_out[f"{prop_folder}_Pred"] = float(np.round(final.ravel()[0], 6))
                row_out[f"{prop_folder}_Std"] = float(np.round(std.ravel()[0], 6))
            except Exception as e:
                row_out[f"{prop_folder}_Pred"] = None
                row_out[f"{prop_folder}_Std"] = None
                row_out[f"{prop_folder}_Error"] = str(e)
        results.append(row_out)

    out_df = pd.DataFrame(results)

    table_style = get_datatable_style(theme)
    table = dash_table.DataTable(
        data=out_df.to_dict('records'),
        columns=[{"name": c, "id": c} for c in out_df.columns],
        **table_style,
        style_table={'overflowX': 'scroll', 'minWidth': '100%'},
        page_size=10
    )

    body = [
        html.Div([
            html.P(f"Processed {len(df)} rows from {filename}.", className='text-muted'),
            dbc.Button(html.Div([
                html.I(className="bi bi-download me-2"),
                "Download output.csv"
            ]), id='btn-download-batch', color='primary', className='mb-3')
        ], className='d-flex justify-content-between align-items-center'),
        html.Div(table),
        # Ensure the download target exists after this section is dynamically updated
        dcc.Download(id='download-batch-csv')
    ]
    # Store CSV in memory as records
    return out_df.to_dict('records'), body


@app.callback(
    Output('download-batch-csv', 'data'),
    Input('btn-download-batch', 'n_clicks'),
    State('batch-pred-store', 'data'),
    prevent_initial_call=True
)
def download_batch(n_clicks, data_records):
    if not data_records:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(data_records)
    return dcc.send_data_frame(df.to_csv, 'output.csv', index=False)


# Theme switch: persist selection and toggle class on root
@app.callback(
    Output('theme-store', 'data'),
    Input('theme-toggle', 'value')
)
def on_theme_toggle(value):
    # value can be [] or [1] for the switch
    is_on = bool(value)
    return 'light' if is_on else 'dark'


@app.callback(
    Output('theme-root', 'className'),
    Input('theme-store', 'data')
)
def update_theme_class(theme):
    # Keep performance class to reduce heavy effects
    return f"theme-{theme} perf-smooth"


@app.callback(
    Output('model-details-output', 'children'),
    Input('model-details-dropdown', 'value'),
    Input('theme-store', 'data')
)
def update_model_details(property_folder, theme):
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
    model_icons = {
        'xgb': 'bi-tree', 'lgb': 'bi-diagram-2', 'cat': 'bi-graph-up',
        'svr': 'bi-vector-pen', 'lin': 'bi-graph-down'
    }
    
    table_style = get_datatable_style(theme)

    for model_name, model in ens.models.items():
        params_dict = {k: v for k, v in model.get_params().items() if v is not None}
        # Limit to most important parameters to avoid clutter
        important_params = dict(list(params_dict.items())[:10])
        params_df = pd.DataFrame(list(important_params.items()), columns=['Parameter', 'Value'])
        
        icon_class = model_icons.get(model_name, 'bi-cpu')
        ensemble_items.append(
            dbc.Card([
                dbc.CardHeader([
                    html.H5(html.Div([
                        html.I(className=f"{icon_class} me-2"),
                        f"🤖 {model_name.upper()} Model"
                    ]), className="mb-0")
                ]),
                dbc.CardBody([
                    html.P(f"Key hyperparameters for the {model_name.upper()} base learner:", 
                           className="text-muted mb-3"),
                    dash_table.DataTable(
                        data=params_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in params_df.columns],
                        **table_style,
                        page_size=8,
                        style_table={'overflowX': 'auto'}
                    )
                ])
            ], className="mt-3 fade-in")
        )

    meta_model_info = {k: v for k, v in ens.meta.get_params().items() if v is not None}
    # Limit meta model params too
    important_meta_params = dict(list(meta_model_info.items())[:8])
    meta_df = pd.DataFrame(list(important_meta_params.items()), columns=['Parameter', 'Value'])
    ensemble_items.append(
        dbc.Card([
            dbc.CardHeader([
                html.H5(html.Div([
                    html.I(className="bi bi-person-gear me-2"),
                    "🎯 Meta-Model (Final Stacker)"
                ]), className="mb-0")
            ]),
            dbc.CardBody([
                html.P("The meta-model learns to optimally combine base learner predictions:", 
                       className="text-muted mb-3"),
                dash_table.DataTable(
                    data=meta_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in meta_df.columns],
                    **table_style,
                    page_size=8,
                    style_table={'overflowX': 'auto'}
                )
            ])
        ], className="mt-3 fade-in")
    )

    return html.Div([
        # ANN Architecture Section
        dbc.Card([
            dbc.CardHeader([
                html.H4(html.Div([
                    html.I(className="bi bi-diagram-3 me-2"),
                    "🧠 Artificial Neural Network (ANN) Architecture"
                ]), className="mb-0")
            ]),
            dbc.CardBody([
                html.P("A sophisticated multi-layer perceptron designed for regression tasks with dropout regularization.", 
                       className="mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Network Topology:", className="fw-bold mb-3"),
                        html.Ul([
                            html.Li(html.Span([html.Strong("Input Layer: "), f"{ann_net[0].in_features} features"])),
                            html.Li(html.Span([html.Strong("Hidden Layer 1 (ReLU): "), f"{ann_net[0].out_features} neurons"])),
                            html.Li(html.Span([html.Strong("Batch Normalization: "), "Applied after first hidden layer"])),
                            html.Li(html.Span([html.Strong("Hidden Layer 2 (ReLU): "), f"{ann_net[4].out_features} neurons"])),
                            html.Li(html.Span([html.Strong("Output Layer: "), "1 neuron (Regression)"]))
                        ], className="list-unstyled")
                    ], md=6),
                    dbc.Col([
                        html.H6("Regularization:", className="fw-bold mb-3"),
                        html.Ul([
                            html.Li("Dropout after first hidden layer"),
                            html.Li("Dropout after second hidden layer"),
                            html.Li("Batch normalization for stable training"),
                            html.Li("ReLU activation for non-linearity")
                        ], className="list-unstyled")
                    ], md=6)
                ])
            ])
        ], className="mb-4 fade-in"),
        
        # Ensemble Architecture Section
        dbc.Card([
            dbc.CardHeader([
                html.H4(html.Div([
                    html.I(className="bi bi-stack me-2"),
                    "🏗️ Ensemble Stack Architecture"
                ]), className="mb-0")
            ]),
            dbc.CardBody([
                html.P("Our ensemble combines 6 diverse base learners using a meta-model approach for optimal performance.", 
                       className="mb-4"),
                html.Div(ensemble_items)
            ])
        ], className="fade-in")
    ], className="fade-in")


# ===== Overview Page interactions =====
@app.callback(
    Output('overview-graphic', 'figure'),
    Input('theme-store', 'data')
)
def render_overview_graph(theme):
    tpl = custom_template_light if theme == 'light' else custom_template
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=90.8,
        number={"suffix": "%", "font": {"size": 56}},
        delta=None,
        title={"text": "Model Confidence"}
    ))
    fig.update_layout(height=300, template=tpl)
    return fig

@app.callback(
    Output('main-tabs', 'active_tab'),
    Input('get-started-btn', 'n_clicks'),
    State('main-tabs', 'active_tab'),
    prevent_initial_call=True
)
def route_from_landing(n_clicks, active):
    if ctx.triggered_id == 'get-started-btn' and n_clicks:
        return 'tab-workbench'
    raise dash.exceptions.PreventUpdate


# ========== RUN APPLICATION ==========
if __name__ == "__main__":

    app.run(debug=True, port=8050)


