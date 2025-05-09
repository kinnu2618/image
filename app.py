from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = load_model('models/imageclassifier.h5')  # adjust path if needed

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Load and preprocess image
        image = Image.open(file).convert('RGB')  # Convert to RGB
        image = image.resize((256, 256))         # Resize to model input
        image_array = np.array(image) / 255.0     # Normalize
        input_tensor = np.expand_dims(image_array, axis=0)

        # Make prediction
        yhat = model.predict(input_tensor)

        # Check output shape and decide label
        if yhat.shape[-1] == 1:  # sigmoid
            confidence = float(yhat[0][0])
            prediction = "Sad 😢" if confidence > 0.5 else "Happy 😊"
        else:  # softmax
            class_names = ["Happy 😊", "Sad 😢"]
            pred_class = np.argmax(yhat[0])
            prediction = class_names[pred_class]
            confidence = float(yhat[0][pred_class])

        return jsonify({'prediction': prediction, 'confidence': round(confidence, 4)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
