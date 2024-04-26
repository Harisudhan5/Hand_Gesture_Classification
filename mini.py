import cv2
import os
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from keras.models import load_model

# Load the pre-trained model
model = load_model("Image_classify.keras")

# Define the categories
category = ['anticlockwise', 'backward', 'clockwise', 'down', 'forward', 'left', 'right', 'up', 'wave']

# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Load the target image for face matching
target_image_path = "Target_Faces/"
target_image = cv2.imread(target_image_path)

# Initialize variables to track face verification result
face_verified = " "

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a black background image with the same dimensions as the frame
    black_bg = np.zeros_like(frame)

    # Process the frame to detect hand landmarks
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    result_hands = hands.process(framergb)
    hand_landmarks = result_hands.multi_hand_landmarks
    
    # Detect faces
    result_faces = face_detection.process(framergb)
    if result_faces.detections:
        for detection in result_faces.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = framergb.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw bounding box around the face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            
            # Perform face matching
            try:
                for file in os.listdir(target_image_path):
                    if file.endswith('jpg'):
                        result = DeepFace.verify(img1_path = os.path.join(target_image_path,file), img2_path = frame, model_name='VGG-Face')
                    if result["verified"]:
                        face_verified = 'tttttt'
                        break
                    else:
                        face_verified = False
            except Exception as e:
                print("Error:", str(e))
                
            # Change text based on face verification result
            if face_verified:
                cv2.putText(frame, face_verified, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
    cv2.imshow("Frame", frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
