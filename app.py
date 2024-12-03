from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__, template_folder='.',static_folder='assets')

# Load the pre-trained model
model = joblib.load('model/smartphone_price_model.pkl')
# Load the pre-trained models
g5_model = joblib.load('model/xgb_classifier_model.pkl')  # Model for predicting 5G capability
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/starter-page')
def home():
    return render_template('starter-page.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    rating = float(request.form['rating'])
    ram_capacity = float(request.form['ram_capacity'])
    internal_memory = float(request.form['internal_memory'])
    resolution_height = int(request.form['resolution_height'])
    resolution_width = int(request.form['resolution_width'])
    refresh_rate = float(request.form['refresh_rate'])

    # Prepare the feature data in the same format as the training data
    input_data = pd.DataFrame({
        'rating': [rating],
        'ram_capacity': [ram_capacity],
        'internal_memory': [internal_memory],
        'resolution_height': [resolution_height],
        'resolution_width': [resolution_width],
        'refresh_rate': [refresh_rate]
    })

    # Predict using the loaded model
    prediction = model.predict(input_data)
    return jsonify({'price': prediction[0]})  # Send the result as JSON

@app.route('/predict_5g', methods=['POST'])
def predict_5g():
    try:
        # Get input values from the form
        rating = float(request.form['rating'])
        ram_capacity = float(request.form['ram_capacity'])
        resolution_height = int(request.form['resolution_height'])
        refresh_rate = float(request.form['refresh_rate'])
        extended_memory_available = int(request.form['extended_memory_available'])

        # Prepare the feature data for 5G prediction
        input_data = pd.DataFrame({
            'rating': [rating],
            'ram_capacity': [ram_capacity],
            'resolution_height': [resolution_height],
            'refresh_rate': [refresh_rate],
            'extended_memory_available': [extended_memory_available],
            'brand_name': ['placeholder_brand'],  # Placeholder if required
            'model': ['placeholder_model'],      # Placeholder if required
            'os': ['placeholder_os']             # Placeholder if required
        })

        # Ensure data types are compatible
        input_data = input_data.astype({
            'rating': float,
            'ram_capacity': float,
            'resolution_height': int,
            'refresh_rate': float,
            'extended_memory_available': int
        })

        # Predict using the loaded 5G model
        g5_prediction = g5_model.predict(input_data, enable_categorical=True)
        g5_result = "Yes" if g5_prediction[0] == 1 else "No"

        return jsonify({'has_5g': g5_result})  # Send the result as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
