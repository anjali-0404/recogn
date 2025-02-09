# from flask import Flask, request, jsonify
# import script  # Import your Python script

# app = Flask(__name__)

# @app.route('/run-python-function', methods=['POST'])
# def run_python_function():
#     input_data = request.json.get('input_data')
#     result = script.some_function(input_data)  # Call your function from script.py
#     return jsonify({"result": result})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load trained sign language model
model_dict = pickle.load(open('./model_A_to_Z.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode Base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)

        # Convert to RGB (for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        x_, y_, data_aux = [], [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
            except Exception:
                predicted_character = "?"

            return jsonify({'prediction': predicted_character})
        else:
            return jsonify({'prediction': "No Hand Detected"})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
