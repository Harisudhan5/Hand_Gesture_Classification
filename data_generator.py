import cv2
import mediapipe as mp
import os
import numpy as np
import time

# Get the parent directory of the current directory
parent_dir = os.path.dirname(os.getcwd())

# Create a folder to save the images in the parent directory's Dataset folder
dataset_dir = os.path.join(parent_dir,'Part 3','Dataset')
try:
    os.makedirs(dataset_dir, exist_ok=True)  # Use exist_ok=True to avoid raising errors if directory already exists
except OSError as e:
    print("Error creating dataset directory:", e)

# Create a folder named "up" inside the Dataset folder
mdir = os.path.join(dataset_dir, 'anticlockwise')
try:
    os.makedirs(mdir, exist_ok=True)
except OSError as e:
    print("Error creating 'up' directory inside Dataset directory:", e)

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create a black background image with the same dimensions as the frame
    black_bg = np.zeros_like(frame)

    # Process the frame to detect hand landmarks
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            # Draw hand landmarks and connections on the black background image
            mp_drawing.draw_landmarks(black_bg, handLMs, mphands.HAND_CONNECTIONS)

    # Generate a unique filename based on current timestamp
    image_path = os.path.join(mdir, 'frame_{}.png'.format(int(time.time()*1000)))  # Multiply by 1000 to get milliseconds

    # Save the image with hand landmarks drawn on it in the "up" folder inside the Dataset folder
    try:
        cv2.imwrite(image_path, black_bg)
        print("Image saved successfully:", image_path)
    except Exception as e:
        print("Error saving image:", e)

    # Display the frame with hand landmarks
    cv2.imshow("Frame", black_bg)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
