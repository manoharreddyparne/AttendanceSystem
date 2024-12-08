import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import face_recognition
from tkinter import messagebox

# Global configurations
TRAINING_IMAGE_PATH = "TrainingImage/"
TRAINING_LABEL_PATH = "TrainingImageLabel/"
STUDENT_DETAILS_PATH = "StudentDetails/student_data.csv"

# Load the trained model
def load_trained_model():
    model_path = f"{TRAINING_LABEL_PATH}/Trainner.yml"
    
    # Check if the model file exists before proceeding
    if not os.path.exists(model_path):
        messagebox.showerror("Error", "Model not found. Please train the model first!")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer

# Function to extract face features using face_recognition
def extract_features(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    return encodings[0] if encodings else None

# Function to test the recognition accuracy
def test_recognition():
    recognizer = load_trained_model()
    if not recognizer:
        return

    df = pd.read_csv(STUDENT_DETAILS_PATH)
    known_face_encodings = []  # Store known face encodings

    # Load and encode known faces from the training set
    for imagePath in os.listdir(TRAINING_IMAGE_PATH):
        img = Image.open(os.path.join(TRAINING_IMAGE_PATH, imagePath))
        img_np = np.array(img)
        face_encoding = extract_features(img_np)
        if face_encoding is not None:
            known_face_encodings.append(face_encoding)

    cam = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, img = cam.read()
        if not ret:
            break  # If frame not captured, break out of the loop

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(img_rgb)

        for (top, right, bottom, left) in face_locations:
            # Extract face encoding for the detected face
            face_encoding = face_recognition.face_encodings(img_rgb, [(top, right, bottom, left)])

            if face_encoding:
                face_encoding = face_encoding[0]

                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True in matches:
                    match_index = matches.index(True)
                    name = df.iloc[match_index]['Name']
                    enrollment = df.iloc[match_index]['Enrollment']

                    # Draw bounding box and label the face
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
                    cv2.putText(img, f"{name} - {enrollment}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Testing Face Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_recognition()
