import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import face_recognition  # Import face_recognition library

# Global configurations
TRAINING_IMAGE_PATH = "TrainingImage/"
TRAINING_LABEL_PATH = "TrainingImageLabel/"
STUDENT_DETAILS_PATH = "StudentDetails/student_data.csv"

# Function to show success messages
def show_success(message):
    print(message)  # This can be changed to messagebox or logging based on your preference

# Function to train the model
def train_model():
    # Check if the model already exists and remove it
    if os.path.exists(f"{TRAINING_LABEL_PATH}/Trainner.yml"):
        os.remove(f"{TRAINING_LABEL_PATH}/Trainner.yml")

    # Create directory if it doesn't exist
    os.makedirs(TRAINING_LABEL_PATH, exist_ok=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Using LBPH model for training
    imagePaths = [os.path.join(TRAINING_IMAGE_PATH, f) for f in os.listdir(TRAINING_IMAGE_PATH)]

    face_samples = []
    ids = []

    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')  # Convert to grayscale ('L' mode)
        imageNp = np.array(img, 'uint8')  # Convert to NumPy array
        
        filename = os.path.basename(imagePath)
        enrollment_id = int(filename.split("_")[0])  # Extract the enrollment ID from filename

        # Use face_recognition to detect faces
        face_locations = face_recognition.face_locations(imageNp)

        for (top, right, bottom, left) in face_locations:
            face_samples.append(imageNp[top:bottom, left:right])  # Store the grayscale face region
            ids.append(enrollment_id)

    recognizer.train(face_samples, np.array(ids))  # Train with grayscale images
    recognizer.save(f"{TRAINING_LABEL_PATH}/Trainner.yml")  # Save the trained model
    show_success("Model trained successfully!")

# Run the training function
train_model()
