from flask import Flask, request, render_template, url_for, redirect
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
import io
import json

# Load your machine learning model and other necessary files
model = pickle.load(open('Crop_Recommendation_Model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Load the plant disease prediction model
try:
    model_disease = tf.keras.models.load_model('trained_model/plant_disease_prediction_model.h5')
except Exception as e:
    print(f"Error loading the plant disease model: {e}")

# Load class indices for the plant disease model
try:
    with open('class_indices.json') as f:
        class_indices = json.load(f)
except Exception as e:
    print(f"Error loading class indices: {e}")

# Dictionary mapping crop index to crop names and image filenames
crop_data = {
    1: {"crop_name": "Rice", "image_filename": "rice.jpg"},
    2: {"crop_name": "Maize", "image_filename": "maize.jpeg"},
    3: {"crop_name": "Jute", "image_filename": "jute.jpg"},
    4: {"crop_name": "Cotton", "image_filename": "cotton.jpeg"},
    5: {"crop_name": "Coconut", "image_filename": "coconut.jpeg"},
    6: {"crop_name": "Papaya", "image_filename": "papaya.jpeg"},
    7: {"crop_name": "Orange", "image_filename": "orange.jpeg"},
    8: {"crop_name": "Apple", "image_filename": "apple.jpeg"},
    9: {"crop_name": "Muskmelon", "image_filename": "muskmelon.jpg"},
    10: {"crop_name": "Watermelon", "image_filename": "watermelon.jpeg"},
    11: {"crop_name": "Grapes", "image_filename": "grapes.jpeg"},
    12: {"crop_name": "Mango", "image_filename": "mango.jpeg"},
    13: {"crop_name": "Banana", "image_filename": "banana.jpg"},
    14: {"crop_name": "Pomegranate", "image_filename": "pomegranate.jpg"},
    15: {"crop_name": "Lentil", "image_filename": "lentil.jpeg"},
    16: {"crop_name": "Blackgram", "image_filename": "blackgram.jpeg"},
    17: {"crop_name": "Mungbean", "image_filename": "mungbean.jpeg"},
    18: {"crop_name": "Mothbeans", "image_filename": "mothbeans.jpeg"},
    19: {"crop_name": "Pigeonpeas", "image_filename": "pigeonpeas.jpg"},
    20: {"crop_name": "Kidneybeams", "image_filename": "kidneybeams.jpeg"},
    21: {"crop_name": "Chickpea", "image_filename": "chickpea.jpeg"},
    22: {"crop_name": "Coffee", "image_filename": "coffee.jpg"},
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template("CropRecommendation.html")

@app.route('/crop_disease_detection')
def crop_disease_detection():
    return render_template("CropDiseaseDetection.html")

@app.route('/fertilizer')
def fertilizer():
    return render_template("Fertilizer.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Input values
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Overall minimum and maximum ranges for all crops (example values)
    overall_ranges = {
        "N": (0, 200),
        "P": (5, 145),
        "K": (5, 205),
        "temp": (10, 44),
        "humidity": (14, 85),
        "ph": (3, 9),
        "rainfall": (20, 300)
    }

    # Validate if input values fall within the overall ranges
    if not (overall_ranges["N"][0] <= N <= overall_ranges["N"][1] and
            overall_ranges["P"][0] <= P <= overall_ranges["P"][1] and
            overall_ranges["K"][0] <= K <= overall_ranges["K"][1] and
            overall_ranges["temp"][0] <= temp <= overall_ranges["temp"][1] and
            overall_ranges["humidity"][0] <= humidity <= overall_ranges["humidity"][1] and
            overall_ranges["ph"][0] <= ph <= overall_ranges["ph"][1] and
            overall_ranges["rainfall"][0] <= rainfall <= overall_ranges["rainfall"][1]):
        result = "The input values are out of the acceptable range."
        return render_template('CropRecommendation.html', result=result)

    # Proceed with prediction if inputs are valid
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    final_features = sc.transform(single_pred)
    prediction = model.predict(final_features)

    if prediction[0] in crop_data:
        crop_name = crop_data[prediction[0]]["crop_name"]
        image_filename = crop_data[prediction[0]]["image_filename"]
        result = f"{crop_name} is the best crop to be cultivated right there"
        return render_template('CropRecommendation.html', result=result, image_filename=image_filename)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        return render_template('CropRecommendation.html', result=result)


@app.route("/upload", methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image with a fixed name
        image_filename = 'uploaded_image.jpg'
        image_path = f'static/{image_filename}'  # Save in the static directory
        file.save(image_path)  # Save the file directly

        def load_and_preprocess_image(image_path, target_size=(224, 224)):
            img = Image.open(image_path)
            img = img.resize(target_size)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.
            return img_array

        def predict_image_class(model, image_path, class_indices):
            preprocessed_img = load_and_preprocess_image(image_path)
            predictions = model.predict(preprocessed_img)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_indices[str(predicted_class_index)]
            return predicted_class_name

        # Get the prediction result
        result = predict_image_class(model_disease, image_path, class_indices)

        

        return render_template('CropDiseaseDetection.html', result=result, image_filename=image_filename)


if __name__ == "__main__":
    app.run(debug=True)
