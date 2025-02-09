import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
TARGET_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z

data = []
labels = []
first_images_with_landmarks = {}  # To store the first processed image with landmarks for each folder

# Process each folder (A-Z)
for dir_ in os.listdir(DATA_DIR):
    if dir_ not in TARGET_CLASSES:
        continue  # Skip non A-Z folders

    folder_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(folder_path):  # Ensure it's a folder
        images = os.listdir(folder_path)
        for idx, img_file in enumerate(images):
            data_aux = []
            x_ = []
            y_ = []

            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}. Skipping.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if idx == 0:
                        mp_drawing.draw_landmarks(
                            img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )

                    # Extract landmarks
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                # Store processed data
                data.append(data_aux)
                labels.append(dir_)

            # Store the first image with landmarks drawn
            if idx == 0 and dir_ not in first_images_with_landmarks:
                first_images_with_landmarks[dir_] = img

# Display the first processed image from each folder
for folder_name, img in first_images_with_landmarks.items():
    img_resized = cv2.resize(img, (600, 600))
    cv2.imshow(f'First Image with Landmarks from {folder_name}', img_resized)

# Wait for a key press to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the processed data
with open('data_A_to_Z.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created and saved as 'data_A_to_Z.pickle' with {len(data)} samples.")
