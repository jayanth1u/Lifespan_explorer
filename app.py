from flask import Flask, jsonify, request, render_template
import pandas as pd
from flask_cors import CORS
import os
import joblib

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load dataset
try:
    df = pd.read_csv("Life Expectancy Data.csv")
    df.dropna(inplace=True)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print("Error: Life Expectancy Data.csv not found!")
    df = pd.DataFrame()

# Load the trained Random Forest model
try:
    random_forest_model = joblib.load('saved_models/Random_Forest.joblib')
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print("Error: Random Forest model file not found!")
    random_forest_model = None

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Get unique countries
@app.route('/countries')
def get_countries():
    try:
        if df.empty:
            return jsonify([])
        countries = sorted(df['Country'].unique().tolist())
        print(f"Returning {len(countries)} countries")
        return jsonify(countries)
    except Exception as e:
        print(f"Error in get_countries: {e}")
        return jsonify([])

# Get data for a specific country
@app.route('/data')
def get_data():
    try:
        country = request.args.get('country')
        if not country or df.empty:
            return jsonify([])

        country_data = df[df['Country'] == country]
        if country_data.empty:
            return jsonify([])

        data = country_data[['Year', 'Life expectancy ', 'GDP', 'Schooling']].copy()
        data = data.sort_values(by='Year')

        records = data.to_dict(orient='records')
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = 0

        print(f"Returning {len(records)} records for {country}")
        return jsonify(records)
    except Exception as e:
        print(f"Error in get_data: {e}")
        return jsonify([])

# Predict life expectancy using Random Forest model
@app.route('/predict', methods=['GET'])
def predict():
    if random_forest_model is None:
        return jsonify({'error': 'Model not loaded, prediction unavailable.'})

    try:
        year = int(request.args.get('year', 2015))
        gdp = float(request.args.get('gdp', 1000))
        schooling = float(request.args.get('schooling', 10))
        features = [[year, gdp, schooling]]

        pred = random_forest_model.predict(features)
        prediction = round(pred[0], 2)

        return jsonify({'Random Forest Prediction': prediction})
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': 'Prediction failed.'})

# Start the Flask app
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='127.0.0.1', port=5000)
