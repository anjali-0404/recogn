import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model (A-Z)
model_dict = pickle.load(open('./model_A_to_Z.p', 'rb'))
model = model_dict['model']

# Open the camera feed
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("Press 'Q' to quit the application.")

while True:
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read from camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            x_ = []
            y_ = []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            # Normalize the landmarks (based on the min values of the landmarks)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Prediction for each hand
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
            except Exception:
                predicted_character = "?"

            # Draw bounding box and prediction for the current hand
            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)
            cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Sign Recognition', frame)

    # Exit on pressing 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
