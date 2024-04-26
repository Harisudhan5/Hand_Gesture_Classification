import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained model
model = load_model("Image_classify.keras")

# Define the categories
category = ['anticlockwise', 'backward', 'clockwise', 'down', 'forward', 'left', 'right', 'up', 'wave']

# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the default camera
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
            
            # Preprocess the image for prediction
            image = cv2.resize(black_bg, (256, 256))  # Resize image to match model input size
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            
            # Predict the class
            predict = model.predict(image)
            class_index = np.argmax(predict)
            confidence = np.max(predict)
            predicted_class = category[class_index]

            # Display the predicted class and confidence score
            text = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame with hand landmarks and predicted class
    cv2.imshow("Frame", framergb)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
