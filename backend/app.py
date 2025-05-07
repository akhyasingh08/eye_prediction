from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
MODEL_PATH = 'model/corneal_ulcer_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    image = preprocess_image(filepath)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    classes = ["No Ulcer", "Micro Punctate", "Macro Punctate", "Coalescent Macro Punctate", "Patch"]
    result = classes[predicted_class]
    
    return jsonify({'prediction': result, 'confidence': float(np.max(prediction))})

if __name__ == '__main__':
    app.run(debug=True)
