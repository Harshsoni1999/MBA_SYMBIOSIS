from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__, template_folder="templates")
CORS(app)

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Path to the dataset folder
DATASET_PATH = "static/"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/routes')
def show_routes():
    return jsonify([str(rule) for rule in app.url_map.iter_rules()])

# Function to load and preprocess dataset
def load_data(crop_name):
    file_path = os.path.join(DATASET_PATH, f"{crop_name}.csv")
    if not os.path.exists(file_path):
        app.logger.error(f"Dataset for {crop_name} not found.")
        return None
    
    df = pd.read_csv(file_path)
    if not all(col in df.columns for col in ["Year", "Month", "WPI", "Rainfall"]):
        app.logger.error(f"Dataset for {crop_name} is missing required columns.")
        return None
    
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
    df.set_index("Date", inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period("M")
    df = df.drop(columns=["Month", "Year"])
    df.loc[:, "WPI"] = df["WPI"].ffill()
    df.loc[:, "Rainfall"] = df["Rainfall"].ffill()
    
    # Add rolling mean features
    df['WPI_MA3'] = df['WPI'].rolling(window=3).mean().fillna(method='bfill')
    df['WPI_MA6'] = df['WPI'].rolling(window=6).mean().fillna(method='bfill')
    df['Rainfall_MA3'] = df['Rainfall'].rolling(window=3).mean().fillna(method='bfill')
    df['Rainfall_MA6'] = df['Rainfall'].rolling(window=6).mean().fillna(method='bfill')
    
    return df

# Function to train SARIMAX model
def train_model(df):
    try:
        target_col = "WPI"
        train = df[df.index.year < 2022]  # Update training data to include years before 2022
        exog_features = ["Rainfall", "WPI_MA3", "WPI_MA6", "Rainfall_MA3", "Rainfall_MA6"]
        
        model = SARIMAX(
            train[target_col],
            exog=train[exog_features],
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit()
        return model_fit, target_col
    except Exception as e:
        app.logger.error(f"Model Training Error: {e}")
        return None, None

# Function to forecast WPI and compare with actual values
def forecast_wpi(model, df, target_col, steps=12):
    test_data = df[df.index.year == 2022]  # Update to predict for 2022
    test_dates = test_data.index.to_timestamp()
    future_dates = test_dates[:steps]  # Ensure alignment with actual test data
    
    future_exog = test_data[["Rainfall", "WPI_MA3", "WPI_MA6", "Rainfall_MA3", "Rainfall_MA6"]].iloc[:steps]
    forecast = model.forecast(steps=steps, exog=future_exog)
    
    results = []
    for i, d in enumerate(future_dates):
        actual_wpi = test_data["WPI"].iloc[i] if i < len(test_data) else None
        actual_rainfall = test_data["Rainfall"].iloc[i] if i < len(test_data) else None
        predicted_wpi = round(forecast[i], 2) if i < len(forecast) else None
        percentage_diff = round(((predicted_wpi - actual_wpi) / actual_wpi) * 100, 2) if actual_wpi else None
        results.append({
            "date": d.strftime("%b %Y"),
            "actual_wpi": actual_wpi,
            "predicted_wpi": predicted_wpi,
            "rainfall": actual_rainfall,
            "percentage_diff": percentage_diff
        })
    return results

@app.route('/forecast/<crop>')
def get_forecast(crop):
    df = load_data(crop)
    if df is None:
        return jsonify({"error": "Crop data not found"}), 404
    
    model, target_col = train_model(df)
    if model is None:
        return jsonify({"error": "Failed to train model"}), 500
    
    return jsonify(forecast_wpi(model, df, target_col))

# Function to generate and return forecast plot
@app.route('/plot/<crop>')
def plot_forecast(crop):
    df = load_data(crop)
    if df is None:
        return jsonify({"error": "Crop data not found"}), 404
    
    model, target_col = train_model(df)
    if model is None:
        return jsonify({"error": "Failed to train model"}), 500
    
    forecast_data = forecast_wpi(model, df, target_col)
    future_dates = [datetime.strptime(d["date"], "%b %Y") for d in forecast_data]
    future_wpi = [d["predicted_wpi"] for d in forecast_data]
    
    plt.figure(figsize=(10, 5))
    plt.plot(df.index.to_timestamp(), df["WPI"], label="Actual WPI", marker="o")
    plt.plot(future_dates, future_wpi, label="Forecast", linestyle="dashed", marker="x")
    plt.xlabel("Date")
    plt.ylabel("WPI")
    plt.legend()
    plt.grid()
    plt.title(f"WPI Forecast for {crop.capitalize()}")
    plot_path = os.path.join("static", f"{crop}_forecast.png")
    plt.savefig(plot_path)
    plt.close()
    
    return jsonify({"image_url": f"/{plot_path}"})

if __name__ == "__main__":
    app.run(debug=True)
