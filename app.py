#import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load the model and preprocessing objects
regmodel = pickle.load(open('property.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

# You'll also need to know which features are categorical and numerical
# Add these based on your original data preparation
num_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF','GrLivArea','FullBath', 'TotRmsAbvGrd ','GarageCars','GarageArea']  
cat_features = ['MSZoning', 'Utilities', 'Neighborhood', 'BldgType','HouseStyle','Heating', 'Electrical', 'SaleType']  

@app.route('/')
def home():
    return render_template('home.html')

def preprocess_input(data_dict):
    """
    Preprocess new input data to match training format
    """
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data_dict])
    
    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df, columns=cat_features, dtype='uint8')
    
    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_columns]
    
    # Scale numerical features
    input_encoded[num_features] = scaler.transform(input_encoded[num_features])
    
    return input_encoded

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print("Raw input data:", data)
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        print("Processed data shape:", processed_data.shape)
        
        # Make prediction
        output = regmodel.predict(processed_data)
        print("Prediction:", output[0])
        
        return jsonify(float(output[0]))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to dictionary
        form_data = request.form.to_dict()
        print("Form data:", form_data)
        
        # Preprocess the input
        processed_data = preprocess_input(form_data)
        print("Processed data shape:", processed_data.shape)
        
        # Make prediction
        output = regmodel.predict(processed_data)[0]
        
        return render_template("home.html", 
                             prediction_text="The House price prediction is ${:,.2f}".format(output))
    
    except Exception as e:
        return render_template("home.html", 
                             prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)