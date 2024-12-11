from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np  # Import numpy here
import pandas as pd  # Import pandas here
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='.',static_folder='assets')

# Load the pre-trained model
model = joblib.load('model/smartphone_price_model_new_last.pkl')

# Charger le modèle et l'encodeur
model_extended_memory = joblib.load('model/random_forest_model.pkl')
encoder = joblib.load('model/encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/starter-page')
def home():
    return render_template('starter-page.html')
# Route to handle prediction
# Load the brand names and their price adjustments from CSV
brand_df = pd.read_csv('brand_name.csv')  # Ensure your CSV is saved as 'brand_name.csv'
brand_price_adjustments = dict(zip(brand_df['brand_name'].str.lower(), brand_df['price_adjustment']))


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
    
    # Return the predicted price as a JSON response
    return jsonify({'price': predicted_price})
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


# Load and preprocess the dataset (same as before)
data = {
    'brand_name': ['oneplus', 'samsung', 'motorola', 'realme', 'xiaomi', 'apple', 'infinix'],
    'price': [6800, 5200, 5600, 5600, 5990, 9000, 7000],
    'ram_capacity': [12, 8, 8, 8, 8, 8, 6],
    'internal_memory': [256, 64, 128, 128, 256, 256, 128],
    'screen_size': [6.7, 6.5, 6.6, 6.7, 6.67, 6.1, 6.5]
}

df = pd.DataFrame(data)
numerical_features = ['price', 'ram_capacity', 'internal_memory', 'screen_size']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Apply one-hot encoding for brand_name
df = pd.get_dummies(df, columns=['brand_name'], drop_first=True)
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user preferences from the request
    user_preference = request.json

    # Normalize the user preferences
    user_vector = pd.DataFrame([[
        user_preference['price'],
        user_preference['ram_capacity'],
        user_preference['internal_memory'],
        user_preference['screen_size']
    ]], columns=numerical_features)

    user_vector = scaler.transform(user_vector)

    # One-hot encode the user input for brand preference (assuming user prefers samsung)
    user_brand = user_preference.get('brand_name', '')
    
    # Make sure the user vector matches the columns of the original df
    user_vector = pd.DataFrame(user_vector, columns=numerical_features)
    
    # Add the brand column to the user vector if the brand is specified
    if user_brand:
        # Create a column for the brand with 1 if the user prefers the given brand, else 0
        for brand in df.columns:
            if brand.startswith(user_brand.lower()):
                user_vector[brand] = 1
            else:
                user_vector[brand] = 0
    else:
        # If no brand is specified, assume user doesn't have a preference
        for brand in df.columns:
            if brand.startswith('brand_name'):
                user_vector[brand] = 0

    # Reorder the columns in user_vector to match df's columns
    user_vector = user_vector[df.columns.difference(['price'])]

    # Calculate cosine similarity
    cos_sim = cosine_similarity(user_vector, df.drop(columns=['price']))
    
    # Add the similarity column to the DataFrame
    df['similarity'] = cos_sim.flatten()

    # Sort the phones by similarity and price
    df_sorted = df.sort_values(by=['similarity', 'price'], ascending=[False, True])

    # Return recommendations as a list of phones
    recommendations = df_sorted[['brand_name', 'price', 'ram_capacity', 'internal_memory', 'screen_size', 'similarity']].to_dict(orient='records')
    return jsonify(recommendations)

@app.route('/predict-extended-memory', methods=['POST'])
def predict_extended_memory():
    try:
        # Retrieve and validate input data
        ram_capacity = request.form.get("ram_capacity", type=float)
        battery_capacity = request.form.get("battery_capacity", type=float)
        internal_memory = request.form.get("internal_memory", type=float)
        processor_brand = request.form.get("processor_brand", type=str)
        price = request.form.get("price", type=float)

        if None in [ram_capacity, battery_capacity, internal_memory, processor_brand, price]:
            return jsonify({"error": "All input fields must be provided and valid."}), 400

        # Encode the processor brand
        try:
            processor_encoded = encoder.transform([processor_brand])[0]
        except Exception as e:
            print(f"Error in encoding processor brand: {e}")  # Log the error
            return jsonify({"error": f"Processor brand encoding failed: {str(e)}"}), 400

        # Create feature array
        features = np.array([[price, ram_capacity, internal_memory, battery_capacity, processor_encoded]])

        # Make prediction
        prediction = model.predict(features)
        prediction_bool = bool(prediction[0])  # Convert to Python boolean

        return jsonify({"extended_memory_available": prediction_bool})

    except Exception as e:
        # Log and handle exceptions
        print(f"Prediction request failed: {e}")  # Log the error
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)