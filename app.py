from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, json, Response, make_response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np  # Import numpy here
import pandas as pd  # Import pandas here
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from flask import Flask, request, jsonify


app = Flask(__name__, template_folder='.',static_folder='assets')


app.secret_key='secretkey'


# Dummy User database (in a real app, you should use a proper database)
# Sample user credentials (in a real app, use a database)
VALID_USERNAME = 'ayoubtoujani'
VALID_PASSWORD = '123'



with open("model/xgboost_5g_model.pkl", "rb") as model_file:
    modelfiveg = pickle.load(model_file)
# Load the pre-trained model
model = joblib.load('model/smartphone_price_model_new_last.pkl')
model_extended_memory = joblib.load('model/random_forest_classifier_last.pkl')


encoder_brand = joblib.load('model/label_encoder_brand_name.pkl')

encoder_os = joblib.load('model/label_encoder_os.pkl')
@app.route('/login', methods=['GET', 'POST'])  # Handle both GET and POST
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the credentials are correct
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True  # Set session to track login state
            flash('Login successful!', 'success')  # Flash success message
            return redirect(url_for('home'))  # Redirect to home after successful login
        else:
            flash('Invalid credentials, please try again.', 'danger')  # Flash error message
            return redirect(url_for('index'))  # Redirect to login page on failure

    # If it's a GET request, render the login form
    return render_template('login.html')
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/starter-page')
def home():
    if not session.get('logged_in'):
        flash('You must log in first.', 'danger')
        return redirect(url_for('index'))  # Redirect to login page if not logged in
    return render_template('starter-page.html')  
# Route to handle prediction
# Load the brand names and their price adjustments from CSV
brand_df = pd.read_csv('brand_name.csv')  # Ensure your CSV is saved as 'brand_name.csv'
brand_price_adjustments = dict(zip(brand_df['brand_name'].str.lower(), brand_df['price_adjustment']))



@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data from POST request
    brand_name = request.form.get('brand_name')
    rating = float(request.form.get('rating'))
    ram_capacity = float(request.form.get('ram_capacity'))
    internal_memory = float(request.form.get('internal_memory'))
    resolution_height = float(request.form.get('resolution_height'))
    resolution_width = float(request.form.get('resolution_width'))
    refresh_rate = float(request.form.get('refresh_rate'))
    battery_capacity = float(request.form.get('battery_capacity'))

    # Encode the brand name using the same encoding as during training
    brand_name_encoding = encode_brand_name(brand_name)  # Assuming you have a function for encoding
    print(f"Encoded brand name for {brand_name}: {brand_name_encoding}")  # Debugging step

    if brand_name_encoding == -1:
        return jsonify({'error': 'Brand name not recognized'})

    # Prepare the input data for prediction (matching the features used during training)
    feature_names = ['brand_name_encoded', 'rating', 'ram_capacity', 'internal_memory',
                     'resolution_height', 'resolution_width', 'refresh_rate','battery_capacity']
    
    input_data = pd.DataFrame([[brand_name_encoding, rating, ram_capacity, internal_memory,
                                resolution_height, resolution_width, refresh_rate,battery_capacity]], columns=feature_names)

    # Check if brand exists in price adjustments
    brand_name_lower = brand_name.lower()
    if brand_name_lower in brand_price_adjustments:
        predicted_price = model.predict(input_data)[0]
        predicted_price *= brand_price_adjustments[brand_name_lower]
        print(f"Adjusted price for {brand_name}: {predicted_price}")  # Debugging step
    else:
        print(f"Brand {brand_name} not found in adjustments!")
        predicted_price = model.predict(input_data)[0]
    
    # Return the predicted price as a JSON response and round to 2 decimal places
    return jsonify({'price': round(predicted_price, 2)-3000})
# Function to encode the brand name (example, you need to adapt this to how you encoded it)
def encode_brand_name(brand_name):
    # Example encoding (you can use LabelEncoder or another method you used during training)
    encoding_map = {
        'apple': 0,
        'asus': 1,
        'blackview': 2,
        'blu': 3,
        'cat': 4,
        'cola': 5,
        'doogee': 6,
        'duoqin': 7,
        'gionee': 8,
        'google': 9,
        'honor': 10,
        'huawei': 11,
        'ikall': 12,
        'infinix': 13,
        'iqoo': 14,
        'itel': 15,
        'jio': 16,
        'lava': 17,
        'leeco': 18,
        'leitz': 19,
        'lenovo': 20,
        'letv': 21,
        'lg': 22,
        'lyf': 23,
        'micromax': 24,
        'motorola': 25,
        'nokia': 26,
        'nothing': 27,
        'nubia': 28,
        'oneplus': 29,
        'oppo': 30,
        'oukitel': 31,
        'poco': 32,
        'realme': 33,
        'redmi': 34,
        'royole': 35,
        'samsung': 36,
        'sharp': 37,
        'sony': 38,
        'tcl': 39,
        'tecno': 40,
        'tesla': 41,
        'vertu': 42,
        'vivo': 43,
        'xiaomi': 44,
        'zte': 45
        }
        
    
    return encoding_map.get(brand_name.lower(), -1)  # Return -1 if brand is not recognized


df = pd.read_csv('data.csv')  # Ensure the file exists
# Convert the dataframe to a list of dictionaries
smartphones = df.to_dict(orient="records")

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Get user preferences from the request
    preferences = request.get_json()
    budget = int(preferences.get('price', 0))
    ram = int(preferences.get('ram_capacity', 0))
    internal_memory = int(preferences.get('internal_memory', 0))
    screen_size = float(preferences.get('screen_size', 0))

    # Filter smartphones based on preferences
    filtered_smartphones = [
        phone for phone in smartphones
        if phone['price'] <= budget and
           phone['ram_capacity'] >= ram and
           (internal_memory == 0 or phone['internal_memory'] >= internal_memory) and
           phone['screen_size'] >= screen_size and 
           phone['brand_name'].strip().lower() == preferences.get('brand_name', phone['brand_name']).strip().lower()
    ]

    # Sort smartphones by RAM, internal memory, screen size, and price
    sorted_smartphones = sorted(
        filtered_smartphones,
        key=lambda x: (-x['ram_capacity'], -x['internal_memory'], -x['screen_size'], x['price'])
    )

    # Add rankings to the sorted list
    ranked_smartphones = [
        {**phone, "rank": idx + 1} for idx, phone in enumerate(sorted_smartphones)
    ]

    # Return the ranked list as a JSON response
    return jsonify(ranked_smartphones)

@app.route('/predict-extended-memory', methods=['POST'])
def predict_extended_memory():
    try:
        # Retrieve and validate input data
        internal_memory = request.form.get("internal_memory", type=float)
        brand_name = request.form.get("brand_name", type=str)
        os = request.form.get("os", type=str)
        ram_capacity = request.form.get("ram_capacity", type=float)

        print(f"Received data: internal_memory={internal_memory}, brand_name={brand_name}, os={os}, ram_capacity={ram_capacity}")

        if None in [internal_memory, brand_name, os, ram_capacity]:
            return jsonify({"error": "All input fields must be provided and valid."}), 400

        # Encode categorical features
        try:
            brand_encoded = encoder_brand.transform([brand_name])[0]
            os_encoded = encoder_os.transform([os])[0]
        except Exception as e:
            print(f"Error in encoding categorical features: {e}")
            return jsonify({"error": f"Encoding failed: {str(e)}"}), 400

        # Create feature array
        features = np.array([[internal_memory, brand_encoded, os_encoded, ram_capacity]])

        # Make prediction
        prediction = model_extended_memory.predict(features)
        prediction_bool = bool(prediction[0])  # Convert to Python boolean

        return jsonify({"extended_memory_available": prediction_bool})

    except Exception as e:
        print(f"Prediction request failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
@app.route('/predict-has5g', methods=['GET'])
def predicthas5g():
    try:
        # Get the query parameters from the URL
        rating = request.args.get('rating', type=float)
        ram_capacity = request.args.get('ram_capacity', type=float)
        refresh_rate = request.args.get('refresh_rate', type=float)
        resolution_height = request.args.get('resolution_height', type=float)
        extended_memory_available = request.args.get('extended_memory_available', type=int)

        # Check if all required parameters are present
        if None in [rating, ram_capacity, refresh_rate, resolution_height, extended_memory_available]:
            return jsonify({"error": "Missing required parameters."}), 400

        # Create a DataFrame with the features to pass to the model
        input_data = pd.DataFrame([{
            'rating': rating,
            'ram_capacity': ram_capacity,
            'refresh_rate': refresh_rate,
            'extended_memory_available': extended_memory_available,
            'resolution_height': resolution_height
        }])

        # Make prediction using the trained model
        prediction =  modelfiveg.predict(input_data)

        # Map the prediction to the corresponding result (0 or 1)
        result = "5G" if prediction[0] == 1 else "No 5G"

        # Return the result as JSON
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Sample prediction logic
def predict_price(features):
    # Example: weights for demo purposes
    weights = [200, 0.8, 1.5, 1.2, 0.5, 0.3]
    return np.dot(features, weights) + 50


# Run the app
if __name__ == '__main__':
    app.run(debug=True)