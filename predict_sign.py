import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import flask
from flask import Flask, request, jsonify
import io
from PIL import Image

app = Flask(__name__)

# Load trained model
model_dict = pickle.load(open('model_A_to_Z.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def predict_sign(image):
    data_aux = []
    x_, y_ = [], []

    H, W, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        try:
            prediction = model.predict([np.asarray(data_aux)])
            return prediction[0]
        except Exception as e:
            print("Prediction error:", e)
            return "?"

    return "No hand detected"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        prediction = predict_sign(image)

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
