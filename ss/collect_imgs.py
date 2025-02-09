import os
import cv2

# Directory where images will be stored
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# List of 26 English letters (A to Z)
classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
dataset_size = 100  # Number of images per letter

# Initialize camera
cap = cv2.VideoCapture(0)  # Adjust the index if the camera doesn't work
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for letter in classes:
    # Create a folder for each letter if it doesn't already exist
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(letter_dir):
        os.makedirs(letter_dir)

    # Check how many images already exist in the folder to append new ones
    existing_images = len(os.listdir(letter_dir))
    if existing_images >= dataset_size:
        print(f"Class '{letter}' already has {dataset_size} images. Skipping...")
        continue

    print(f'Collecting data for letter: {letter} (Press "Q" to start capturing images)')

    # Wait for user input to start capturing images for this letter
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        cv2.putText(frame, f'Ready for {letter}? Press "Q"!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = existing_images  # Continue from where the folder count left off
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        # Save the image, appending new images to the existing ones
        cv2.imwrite(os.path.join(letter_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
