import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# --- DATA PREPARATION ---
def load_and_prep_data():
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    
    # Feature Engineering for Modeling
    le_dict = {}
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
    return df, df_encoded, le_dict

df, df_encoded, encoders = load_and_prep_data()

# --- MODEL TRAINING (Simple Forecasting/Prediction) ---
X = df_encoded[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
y = df_encoded['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- VISUALIZATION FUNCTIONS ---
def get_stats():
    return df.describe().reset_index()

def plot_price_distribution():
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['selling_price'], kde=True, color='#2c3e50', ax=ax)
    ax.set_title("Distribution of Selling Prices", fontsize=14)
    plt.tight_layout()
    return fig

def plot_categorical_analysis(feature):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=feature, y='selling_price', data=df, palette='viridis', ax=ax)
    ax.set_title(f"Price Analysis by {feature.capitalize()}", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def predict_price(year, km, fuel, seller, trans, owner):
    # Encode inputs
    input_data = pd.DataFrame([[
        year, km, 
        encoders['fuel'].transform([fuel])[0],
        encoders['seller_type'].transform([seller])[0],
        encoders['transmission'].transform([trans])[0],
        encoders['owner'].transform([owner])[0]
    ]], columns=['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])
    
    prediction = model.predict(input_data)[0]
    return f"₹ {round(prediction, 2):,}"

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 CarDekho Inventory & Valuation Insights")
    gr.Markdown("### Professional Analytics Dashboard & Price Forecasting Tool")
    
    with gr.Tab("Statistical Overview"):
        gr.Markdown("#### Key Dataset Metrics")
        gr.DataFrame(get_stats)
        
        with gr.Row():
            plot_dist = gr.Plot(value=plot_price_distribution(), label="Price Spread")
            with gr.Column():
                feat_selector = gr.Dropdown(
                    choices=['fuel', 'seller_type', 'transmission', 'owner'], 
                    value='fuel', 
                    label="Select Feature for Comparative Analysis"
                )
                plot_cat = gr.Plot(label="Categorical Breakdown")
        
        feat_selector.change(plot_categorical_analysis, inputs=feat_selector, outputs=plot_cat)

    with gr.Tab("Market Forecasting"):
        gr.Markdown("#### Estimated Selling Price Predictor")
        with gr.Row():
            with gr.Column():
                year_in = gr.Number(label="Year of Manufacture", value=2015)
                km_in = gr.Number(label="Kilometers Driven", value=50000)
                fuel_in = gr.Dropdown(choices=list(df['fuel'].unique()), label="Fuel Type")
            with gr.Column():
                seller_in = gr.Dropdown(choices=list(df['seller_type'].unique()), label="Seller Type")
                trans_in = gr.Dropdown(choices=list(df['transmission'].unique()), label="Transmission")
                owner_in = gr.Dropdown(choices=list(df['owner'].unique()), label="Owner History")
        
        predict_btn = gr.Button("Calculate Estimated Market Value", variant="primary")
        output_price = gr.Textbox(label="Predicted Selling Price (INR)", placeholder="Result will appear here...")
        
        predict_btn.click(
            predict_price, 
            inputs=[year_in, km_in, fuel_in, seller_in, trans_in, owner_in], 
            outputs=output_price
        )

# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch()