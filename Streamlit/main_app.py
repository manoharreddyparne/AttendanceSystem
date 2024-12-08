import csv
import os
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Function to get the next serial number for the image files
def get_next_serial_number(folder_path, enrollment, name):
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return 1
    serial_numbers = [int(f.split('_')[0]) for f in files if f.split('_')[0].isdigit()]
    return max(serial_numbers) + 1 if serial_numbers else 1

# Function to save the uploaded file (only if it passes the checks)
def save_uploaded_file(uploadedfile, name, enrollment):
    try:
        if not os.path.exists("TrainingImage"):
            os.makedirs("TrainingImage")
        folder_path = "TrainingImage"
        serial_number = get_next_serial_number(folder_path, enrollment, name)
        filename = f"{enrollment}_{name}_{serial_number}.jpg"

        # Process the image to verify if a face is detected
        face_image = process_selfie_image(uploadedfile)
        if face_image is not None:
            # Check if the file with the same name already exists
            if os.path.exists(os.path.join(folder_path, filename)):
                st.error(f"File already exists: {filename}. Image not saved.")
                return None  # Don't save if the file already exists

            # Save the cropped face image
            cv2.imwrite(os.path.join(folder_path, filename), face_image)
            # Save student info to the CSV file
            save_student_data(enrollment, name)
            return os.path.join(folder_path, filename)
        else:
            st.error("No face detected in the uploaded image!")
            return None
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

# Function to process the selfie image and predict (extract the face)
def process_selfie_image(uploadedfile):
    try:
        # Convert uploaded image to numpy array
        img = Image.open(uploadedfile)
        img = np.array(img)

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained face detector
        face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None  # No faces detected
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]  # Crop the face region
            return face  # Return the detected face region
        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Function to save the student details to the CSV file
def save_student_data(enrollment, name):
    # Check if the student already exists in the CSV file
    if check_duplicate_registration(enrollment):
        st.error(f"Enrollment {enrollment} already exists. Cannot add.")
        return

    try:
        with open("StudentDetails/student_data.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([enrollment, name])
        st.success(f"Student {name} with Enrollment {enrollment} added successfully.")
    except Exception as e:
        st.error(f"Error saving student data: {str(e)}")

# Function to check if enrollment exists in the student_data.csv
def check_duplicate_registration(enrollment):
    try:
        if not os.path.exists("StudentDetails/student_data.csv"):
            return False
        df = pd.read_csv("StudentDetails/student_data.csv")
        return enrollment in df['Enrollment'].values
    except Exception as e:
        st.error(f"Error checking duplicate: {str(e)}")
        return False

# Streamlit interface
st.title("Selfie Upload for Attendance System")

# Input fields for name and enrollment number
name = st.text_input("Enter your name:")
enrollment = st.text_input("Enter your enrollment number:")

# Image upload widget
uploaded_image = st.file_uploader("Upload your selfie image", type=["jpg", "jpeg", "png"])

# Flag to indicate whether the file has been successfully saved
if "is_saved" not in st.session_state:
    st.session_state.is_saved = False  # Initialize the flag in session state

# Image processing buffer and save logic
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Show loading indicator while processing
    with st.spinner("Processing... Please wait."):

        # Process the image to detect face
        face_image = process_selfie_image(uploaded_image)

        # If face is found, show the 'Save Image' button
        if face_image is not None:
            st.image(face_image, caption="Processed Face", use_container_width=True)
            # Only show the save button once the face is processed
            if st.button("Save Image"):
                if name and enrollment:
                    file_path = save_uploaded_file(uploaded_image, name, enrollment)
                    if file_path:
                        st.success(f"Image saved successfully: {file_path}")
                        st.session_state.is_saved = True  # Set flag to true once saved
                    else:
                        st.error("Failed to save the image.")
                else:
                    st.error("Please provide both name and enrollment number.")
        else:
            st.error("No face detected. Please upload an image with a visible face.")
    
    # Show the "Add New Entry" button only after successful save
    if st.session_state.is_saved:
        st.success("Entry successfully saved. Ready for a new entry.")
        st.session_state.is_saved = False  # Reset flag for a fresh start
