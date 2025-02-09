import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the trained model
try:
    model_dict = pickle.load(open('./model_A_to_Z.p', 'rb'))
    model = model_dict['model']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def process_image_for_prediction(image_path):
    try:
        # Read image using OpenCV
        frame = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape

        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return {"error": "No hand detected in the image"}

        # Process the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract and normalize landmarks
        x_ = []
        y_ = []
        data_aux = []

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = prediction[0]

        return {
            "status": "success",
            "prediction": predicted_character,
            "hand_position": {
                "x1": int(min(x_) * W),
                "y1": int(min(y_) * H),
                "x2": int(max(x_) * W),
                "y2": int(max(y_) * H)
            }
        }

    except Exception as e:
        logger.error(f"Error processing image for prediction: {str(e)}")
        return {"error": str(e)}

def is_valid_image(file_stream):
    try:
        image = Image.open(file_stream)
        image.verify()
        file_stream.seek(0)
        return True
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "Server is running. Use /predict endpoint for sign language recognition."
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    logger.debug(f"Headers: {dict(request.headers)}")

    try:
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({
                "status": "error",
                "message": "No image file in request"
            }), 400

        file = request.files['image']
        
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({
                "status": "error",
                "message": "No selected file"
            }), 400

        if not is_valid_image(file):
            logger.error("Invalid image file")
            return jsonify({
                "status": "error",
                "message": "Invalid image file. Please upload a valid image."
            }), 400

        # Generate unique filename and save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.seek(0)
        image = Image.open(file)
        image = image.convert('RGB')
        image.save(filepath, 'JPEG', quality=85)
        
        logger.info(f"Successfully saved image to {filepath}")

        # Process the image for prediction
        prediction_result = process_image_for_prediction(filepath)
        
        if "error" in prediction_result:
            return jsonify({
                "status": "error",
                "message": prediction_result["error"]
            }), 400

        return jsonify({
            "status": "success",
            "message": "Image processed successfully!",
            "data": {
                "filename": filename,
                "prediction": prediction_result["prediction"],
                "hand_position": prediction_result["hand_position"]
            }
        }), 200

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)