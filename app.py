# =============================================================================
# MSc Business Analytics Final Project
# Standalone Interactive Dashboard using Dash and Plotly
# =============================================================================

# --- 1. SETUP AND IMPORTS ---
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import os
import joblib # For loading the pre-trained model
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import dash
from dash import dcc, html

warnings.filterwarnings('ignore')

# --- 2. DATA LOADING AND PREPARATION ---

print("Loading and preparing data for the dashboard...")

# Dash requires a folder named 'assets' for static files
if not os.path.exists('assets'):
    os.makedirs('assets')

try:
    listings_df = pd.read_csv('listings.csv')
    calendar_df = pd.read_csv('calendar.csv')
    df_clean = pd.read_csv('final_airbnb_data.csv')
    df_clean['date'] = pd.to_datetime(df_clean['date'])
except FileNotFoundError:
    print('Error: Ensure listings.csv, calendar.csv, and final_airbnb_data.csv are present.')
    exit()

# --- Data for Visual 1: Demand by Neighbourhood ---
merged_df_eda = pd.merge(calendar_df[['listing_id', 'available']], listings_df[['id', 'neighbourhood_cleansed']], left_on='listing_id', right_on='id')
merged_df_eda['is_booked'] = np.where(merged_df_eda['available'] == 'f', 1, 0)
top_10_neighbourhoods = listings_df['neighbourhood_cleansed'].value_counts().iloc[:10].index
top_10_data = merged_df_eda[merged_df_eda['neighbourhood_cleansed'].isin(top_10_neighbourhoods)]
demand_by_neighbourhood = top_10_data.groupby('neighbourhood_cleansed')['is_booked'].mean().sort_values(ascending=True) * 100

# --- Data for Visual 2: Host Distribution ---
host_counts = listings_df['host_id'].value_counts()
num_single_listing_hosts = (host_counts == 1).sum()
num_two_listing_hosts = (host_counts == 2).sum()
num_three_to_five_hosts = ((host_counts >= 3) & (host_counts <= 5)).sum()
num_six_plus_hosts = (host_counts > 5).sum()
host_distribution_data = {
    'labels': ['1 Listing', '2 Listings', '3-5 Listings', '6+ Listings'],
    'data': [int(num_single_listing_hosts), int(num_two_listing_hosts), int(num_three_to_five_hosts), int(num_six_plus_hosts)]
}

# --- Data for Financial Simulation & Model Loading ---
features_to_exclude = ['is_booked', 'listing_id', 'date']
split_date = '2025-12-01'
train_df = df_clean[df_clean['date'] < split_date] # Still needed for base prices
test_df = df_clean.loc[df_clean['date'] >= split_date].copy()
X_test = test_df.drop(columns=features_to_exclude)
y_test = test_df['is_booked']

# Load the pre-trained model instead of training it
print("Loading pre-trained model from champion_model.joblib...")
champion_model = joblib.load('champion_model.joblib')
print("Model loaded.")

# Financial Simulation Results
test_df['predicted_booking_prob'] = champion_model.predict_proba(X_test)[:, 1]
pricing_multipliers = {'Very Low Demand': 0.85, 'Low Demand': 1.00, 'Medium Demand': 1.15, 'High Demand': 1.40, 'Peak Demand': 1.80}
prob_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
prob_labels = ['Very Low Demand', 'Low Demand', 'Medium Demand', 'High Demand', 'Peak Demand']
test_df['demand_tier'] = pd.cut(test_df['predicted_booking_prob'], bins=prob_bins, labels=prob_labels, include_lowest=True)
base_prices = train_df.groupby('listing_id')['price'].median().reset_index()
base_prices.rename(columns={'price': 'base_price'}, inplace=True)
test_df = pd.merge(test_df, base_prices, on='listing_id', how='left')
test_df['base_price'].fillna(train_df['price'].median(), inplace=True)
test_df['dynamic_price'] = test_df.apply(lambda row: row['base_price'] * pricing_multipliers.get(row['demand_tier'], 1.0), axis=1)

# Add Host's Listed Price for comparison
host_listed_price_df = listings_df[['id', 'price']].copy()
host_listed_price_df.rename(columns={'id': 'listing_id', 'price': 'host_listed_price'}, inplace=True)
host_listed_price_df['host_listed_price'] = host_listed_price_df['host_listed_price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)
test_df = pd.merge(test_df, host_listed_price_df, on='listing_id', how='left')
test_df['host_listed_price'].fillna(test_df['base_price'], inplace=True)

dynamic_revenue = (test_df['dynamic_price'] * y_test).sum()
static_revenue = (test_df['base_price'] * y_test).sum()
host_listed_revenue = (test_df['host_listed_price'] * y_test).sum()

percentage_uplift_vs_static = ((dynamic_revenue - static_revenue) / static_revenue) * 100
percentage_uplift_vs_host = ((dynamic_revenue - host_listed_revenue) / host_listed_revenue) * 100

# Robustness Check Results
from sklearn.metrics import roc_auc_score
original_auc = roc_auc_score(y_test, test_df['predicted_booking_prob'])
y_test_perturbed = y_test.copy()
booked_indices = y_test_perturbed[y_test_perturbed == 1].index
num_to_flip = int(len(booked_indices) * 0.10)
indices_to_flip = np.random.choice(booked_indices, size=num_to_flip, replace=False)
y_test_perturbed.loc[indices_to_flip] = 0
perturbed_auc = roc_auc_score(y_test_perturbed, test_df['predicted_booking_prob'])
performance_degradation = ((original_auc - perturbed_auc) / original_auc) * 100

print("Data preparation complete.")

# --- 3. GENERATE SHAP PLOTS AS IMAGES ---
# This section is intentionally commented out for deployment.
# The server will use the pre-generated images from the 'assets' folder.

# --- 4. CREATE PLOTLY FIGURES ---
print("Creating Plotly figures...")

# Figure 1: Demand by Neighbourhood
fig1 = go.Figure(go.Bar(
    x=demand_by_neighbourhood.values,
    y=demand_by_neighbourhood.index,
    orientation='h',
    marker_color='#4f46e5'
))
fig1.update_layout(title_text='<b>Visual 1: Where is the Demand?</b>', yaxis_title='Neighbourhood', xaxis_title='Average Booking Rate (%)', xaxis_range=[0,100], template='plotly_white', font=dict(family="Georgia, serif", size=12))

# Figure 2: Host Distribution
fig2 = go.Figure(go.Bar(
    x=host_distribution_data['labels'],
    y=host_distribution_data['data'],
    marker_color='#14b8a6'
))
fig2.update_layout(title_text='<b>Visual 2: Who Are the Hosts?</b>', yaxis_type="log", xaxis_title='Host Category', yaxis_title='Number of Hosts (Log Scale)', template='plotly_white', font=dict(family="Georgia, serif", size=12))

# Figure 6: Financial Uplift
fig6 = go.Figure(go.Bar(
    x=["Host's Listed Price", "Static Average Price", "Dynamic Price"],
    y=[host_listed_revenue, static_revenue, dynamic_revenue],
    text=[f"£{host_listed_revenue/1e6:.2f}M", f"£{static_revenue/1e6:.2f}M", f"£{dynamic_revenue/1e6:.2f}M"],
    textposition='auto',
    marker_color=['#ef4444', '#3b82f6', '#14b8a6']
))
fig6.update_layout(title_text='<b>Visual 6: The Financial Uplift</b>', yaxis_title='Total Revenue (£)', template='plotly_white', font=dict(family="Georgia, serif", size=12))

print("Plotly figures created.")

# --- 5. BUILD THE DASH APP ---
print("Building Dash app...")

# Initialize the app
app = dash.Dash(__name__, external_scripts=['https://cdn.tailwindcss.com'])
server = app.server # Expose server for deployment

# App Layout
app.layout = html.Div(className="bg-gray-100 text-gray-800 p-4 sm:p-6 md:p-8", style={'fontFamily': 'Georgia, serif', 'fontSize': '12pt'}, children=[
    html.Div(className="max-w-7xl mx-auto", children=[
        # Header
        html.Header(className="mb-8", children=[
            html.H1("A Smart Pricing Framework for Airbnb Hosts", className="text-3xl md:text-4xl font-bold text-gray-900", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '16pt'}),
            html.P("A Data-Driven Approach to Solving Algorithmic Inequality in Bristol", className="text-lg text-gray-600 mt-2")
        ]),

        # Main Grid
        html.Main(className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6", children=[
            # Section 1
            html.Div(className="md:col-span-2 lg:col-span-3 bg-white p-6 rounded-xl shadow-md", children=[
                html.H2("Section 1: The Bristol Airbnb Market", className="text-2xl font-bold mb-4 border-b pb-2", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '16pt'}),
                html.Div(className="grid grid-cols-1 lg:grid-cols-2 gap-6", children=[
                    dcc.Graph(figure=fig1, id='demand-by-neighbourhood-graph', config={'displayModeBar': False}),
                    dcc.Graph(figure=fig2, id='host-distribution-graph', config={'displayModeBar': False})
                ])
            ]),
            # Section 2
            html.Div(className="lg:col-span-1 bg-white p-6 rounded-xl shadow-md", children=[
                html.H2("Section 2: The Solution", className="text-2xl font-bold mb-4 border-b pb-2", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '16pt'}),
                html.H3("Model Performance", className="text-xl font-semibold mb-2", style={'fontFamily': 'Inter, sans-serif'}),
                html.P("The XGBoost model was selected as the champion, demonstrating strong predictive accuracy.", className="text-gray-600 mb-4"),
                html.Div(className="space-y-3", children=[
                    html.Div([html.P("Champion Model", className="text-sm text-blue-700 font-medium"), html.P("XGBoost", className="text-2xl font-bold text-blue-900", style={'fontFamily': 'Inter, sans-serif'})], className="bg-blue-50 p-4 rounded-lg"),
                    html.Div([html.P("AUC-ROC Score", className="text-sm text-green-700 font-medium"), html.P(f"{original_auc:.4f}", className="text-2xl font-bold text-green-900", style={'fontFamily': 'Inter, sans-serif'})], className="bg-green-50 p-4 rounded-lg")
                ])
            ]),
            # Section 3
            html.Div(className="md:col-span-2 bg-white p-6 rounded-xl shadow-md", children=[
                html.H2("Section 3: Why Does the Model Work?", className="text-2xl font-bold mb-4 border-b pb-2", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '16pt'}),
                html.Div(className="grid grid-cols-1 xl:grid-cols-2 gap-6", children=[
                    html.Div([html.H3("Key Drivers of Demand", className="text-xl font-semibold mb-2", style={'fontFamily': 'Inter, sans-serif'}), html.Img(src=app.get_asset_url('shap_bar_plot.png'), className="rounded-lg w-full", alt="SHAP bar chart showing feature importance. Top features are beds, review recency, and month.")]),
                    html.Div([html.H3("How Drivers Work", className="text-xl font-semibold mb-2", style={'fontFamily': 'Inter, sans-serif'}), html.Img(src=app.get_asset_url('shap_beeswarm_plot.png'), className="rounded-lg w-full", alt="SHAP beeswarm plot showing low prices and recent reviews increase booking probability.")])
                ])
            ]),
            # Section 4
            html.Div(className="md:col-span-2 lg:col-span-3 bg-white p-6 rounded-xl shadow-md", children=[
                html.H2("Section 4: Quantifying the ROI", className="text-2xl font-bold mb-4 border-b pb-2", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '16pt'}),
                html.Div(className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center", children=[
                    html.Div(dcc.Graph(figure=fig6, id='revenue-uplift-graph', config={'displayModeBar': False}), className="lg:col-span-2"),
                    html.Div(className="lg:col-span-1 space-y-4", children=[
                        html.H3("ROI Scorecard", className="text-xl font-semibold mb-2 text-center", style={'fontFamily': 'Inter, sans-serif'}),
                        html.Div([html.P("Uplift vs. Host's Price", className="text-sm text-red-700 font-medium"), html.P(f"{percentage_uplift_vs_host:.2f}%", className="text-3xl font-bold text-red-900", style={'fontFamily': 'Inter, sans-serif'})], className="bg-red-50 p-4 rounded-lg text-center"),
                        html.Div([html.P("Uplift vs. Static Average", className="text-sm text-teal-700 font-medium"), html.P(f"{percentage_uplift_vs_static:.2f}%", className="text-3xl font-bold text-teal-900", style={'fontFamily': 'Inter, sans-serif'})], className="bg-teal-50 p-4 rounded-lg text-center")
                    ])
                ])
            ]),
            # Section 5
            html.Div(className="md:col-span-2 lg:col-span-3 bg-white p-6 rounded-xl shadow-md", children=[
                html.H2("Section 5: Final Checks", className="text-2xl font-bold mb-4 border-b pb-2", style={'fontFamily': 'Inter, sans-serif', 'fontSize': '16pt'}),
                html.Div(className="flex items-center justify-center text-center", children=[
                    html.Div([
                        html.H3("Robustness Check", className="text-xl font-semibold mb-2", style={'fontFamily': 'Inter, sans-serif'}),
                        html.P("When a 10% error rate was introduced, model performance only degraded by:", className="text-gray-600 mb-4"),
                        html.Div(html.P(f"{performance_degradation:.2f}%", className="text-4xl font-bold text-red-900", style={'fontFamily': 'Inter, sans-serif'}), className="bg-red-50 p-6 rounded-full inline-block")
                    ])
                ])
            ])
        ])
    ])
])

# --- 6. RUN THE SERVER LOCALLY ---
if __name__ == '__main__':
    app.run_server(debug=True)