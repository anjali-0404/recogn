# from flask import Flask, request, jsonify
# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
# import io
# from PIL import Image

# app = Flask(__name__)

# # Load the trained model
# model_dict = pickle.load(open('./model_A_to_Z.p', 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe Hands module
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         image_data = data['image']
        
#         # Decode base64 image
#         image_bytes = base64.b64decode(image_data)
#         image = Image.open(io.BytesIO(image_bytes))
#         frame = np.array(image)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # Process the image with MediaPipe
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_, y_, data_aux = [], [], []

#                 for lm in hand_landmarks.landmark:
#                     x_.append(lm.x)
#                     y_.append(lm.y)

#                 for lm in hand_landmarks.landmark:
#                     data_aux.append(lm.x - min(x_))
#                     data_aux.append(lm.y - min(y_))

#                 try:
#                     prediction = model.predict([np.asarray(data_aux)])
#                     predicted_character = prediction[0]
#                 except Exception:
#                     predicted_character = "?"

#                 return jsonify({'prediction': predicted_character})
        
#         return jsonify({'prediction': "No hand detected"})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
import base64
import logging
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_image(base64_image):
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame.shape

        # Process with MediaPipe
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return {"error": "No hand detected"}

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
            "prediction": predicted_character,
            "confidence": "high",  # You could add actual confidence scores if your model provides them
            "hand_position": {
                "x1": int(min(x_) * W),
                "y1": int(min(y_) * H),
                "x2": int(max(x_) * W),
                "y2": int(max(y_) * H)
            }
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400

    base64_image = request.json['image']
    
    try:
        result = process_image(base64_image)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)