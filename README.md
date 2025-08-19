# 🚀 EcoFuelFusion.ai — Interactive Fuel Blend Ensemble Predictor  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![Dash](https://img.shields.io/badge/Dash-Framework-0098ff.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-ML-orange.svg)  
![License](https://img.shields.io/badge/License-Research%20Use-lightgrey.svg)  
![Status](https://img.shields.io/badge/Status-Active-success.svg)  

Accurate, explainable **fuel blend property predictions** powered by a stacked ensemble of Machine Learning models and an immersive **Dash web app** with **dark/light themes**, **glassmorphism styling**, and **animated UI interactions**.  

---

## 📑 Table of Contents
1. [Approach Summary](#-approach-summary)  
2. [Feature Engineering](#-feature-engineering)  
3. [App Features](#-app-features)  
4. [Repository Structure](#-repository-structure)  
5. [Theming & UI Enhancements](#-theming--ui-enhancements)  
6. [Tools & Libraries](#-tools-and-libraries)  
7. [Getting Started](#-getting-started)  
8. [Deployment](#-deployment-planned-on-render)  
9. [Input Format](#-input-format)  
10. [Output](#-output)  
11. [Credits & Contributors](#-credits--contributors)  
12. [License](#-license)  

---

## 🧠 Approach Summary  

The predictor forecasts **10 blend properties** for user-defined or uploaded fuel blends using a **stacked ensemble regression pipeline**.  

- **Base Models**  
  - ANN (PyTorch)  
  - XGBoost  
  - LightGBM  
  - CatBoost  
  - SVR  
  - Linear Regression  

- **Meta Model**  
  - GradientBoostingRegressor (scikit-learn) to optimally combine outputs  

- **Optimization**  
  - All models tuned with **Optuna**  

---

## ⚙️ Feature Engineering  

For each property (`BlendProperty1` → `BlendProperty10`):  

- **Weighted Properties**  

Weighted\_Property\_i = Σ (Component\_{j}\_fraction × Component\_{j}\_Property\_{i}), j = 1..5


- **Statistical Features**  
  - Mean across 5 components  
  - Standard deviation across 5 components  

---

## 🖥️ App Features  

The Dash application provides an **interactive multi-tab UI**:  

1. **🏠 Overview** – Introduction & model confidence KPI  
2. **🔧 Prediction Workbench** – Upload CSV or configure manually, run predictions, visualize blend composition, batch support  
3. **🔬 Model Details** – Explore ensemble architecture, ANN layers, hyperparameters  
4. **💡 Layman’s Guide** – Simplified, step-by-step pipeline explanation with icons  
5. **⚙️ Technical Details** – Full ML workflow explanation (feature engineering → base models → meta-model)  
6. **📊 Data Analytics (EDA)** – Correlations, histograms, scatter/box plots  

---

## 📁 Repository Structure  

```
EcoFuelFusion.ai/
│
├── app.py                            # Main Dash application
│
├── models/                           # Pre-trained model folders
│   ├── BlendProperty1/
│   │   ├── x_scaler.pkl
│   │   ├── y_scaler.pkl
│   │   ├── xgb.pkl, lgb.pkl, svr.pkl, lin.pkl
│   │   ├── cat.cbm
│   │   ├── ann.pt
│   │   └── meta.pkl
│   ├── BlendProperty2/
│   └── ... BlendProperty10/
│
├── data/
│   └── original_data.csv             # Dataset for EDA
│
├── assets/                           # Static assets (auto-loaded by Dash)
│   ├── style.css                     # Custom CSS (dark/light themes, glassmorphism)
│   ├── scroll.js                     # JS enhancements (smooth scroll, reveal animations)
│   ├── tech_step*.png                 # Pipeline illustrations
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🎨 Theming & UI Enhancements  

- **Dark/Light themes** with toggle  
- **Glassmorphism cards, alerts, tables, navbars** (`style.css`)  
- **Smooth scrolling & reveal animations** (`scroll.js`)  
- **Modernized inputs**: sliders, dropdowns, accordion-based input forms  

---

## 🛠️ Tools and Libraries  

| Category      | Libraries / Tools                                                              |
| ------------- | ------------------------------------------------------------------------------ |
| Web Framework | `Dash`, `dash-bootstrap-components`, `Plotly`                                  |
| ML Models     | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `SVR`                       |
| Deep Learning | `PyTorch`                                                                      |
| Optimization  | `Optuna`                                                                       |
| Data Handling | `pandas`, `numpy`                                                              |
| Deployment    | [Google Cloud](https://console.cloud.google.com/)                                       |
| Frontend      | `CSS` (themes, glassmorphism), `JavaScript` (scrolling & animations)           |

---

## 🚀 Getting Started  

### ✅ Prerequisites  
- Python 3.10+ (tested on 3.13.5)  
- Virtual environment recommended  

### 📥 Step 1: Clone Repo  
```bash
git clone https://github.com/ThePrachiShuk/EcoFuelFusion
cd EcoFuelFusion.ai
```

### 📦 Step 2: Install Dependencies  
```bash
pip install -r requirements.txt
```

### 🏃 Step 3: Run the App  
```bash
python app.py
```
Then open: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)  

---

<!--## 🌐 Deployment (Planned on Render)  
Deployment instructions will be added once configuration is finalized.  

---
-->

## 📈 Input Format  

- **Manual Input**: Configure via sliders & number fields  
- **CSV Upload**:  
  - Headerless CSV, single row  
  - First 5 columns → component fractions (%)  
  - Next 50 columns → component properties (5×10)  

---

## 📊 Output  

- Ensemble predictions for 10 blend properties  
- Per-property uncertainty (std deviation across base models)  
- Model Confidence KPI card  
- Interactive visualizations (composition + EDA)  

---

## 🤝 Credits & Contributors  

- [ ] Arnav Bansal - [TytonTerrapin](https://github.com/TytonTerrapin)
- [ ] Nakul Tanwar - [Nakul-28](https://github.com/Nakul-28)   
- [ ] Prachetas Shukla - [ThePrachiShuk](https://github.com/ThePrachiShuk)
- [ ] Ranbir Singh - [Raj-2006-afk](https://github.com/Raj-2006-afk)     



---

## ⚖️ License  

© 2025 **EcoFuelFusion.ai**<br>
All rights reserved. This project is for educational and research purposes only.<br>
Unauthorized copying, modification, or distribution of this project is prohibited without explicit permission.
