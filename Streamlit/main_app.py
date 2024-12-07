import os
import cv2
import streamlit as st
from PIL import Image

# Function to get the next serial number for the image files
def get_next_serial_number(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return 1
    serial_numbers = [int(f.split('_')[0]) for f in files if f.split('_')[0].isdigit()]
    return max(serial_numbers) + 1 if serial_numbers else 1

# Function to save the uploaded selfie image with specified naming format
def save_uploaded_file(uploadedfile, name, enrollment):
    try:
        if not os.path.exists("Selfie_Images"):
            os.makedirs("Selfie_Images")
        folder_path = "Selfie_Images"
        serial_number = get_next_serial_number(folder_path)
        filename = f"{serial_number}_{enrollment}_{name}.jpg"
        img = Image.open(uploadedfile)
        img.save(os.path.join(folder_path, filename))
        return os.path.join(folder_path, filename)
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

# Function to process the selfie image and predict
def process_selfie_image(image_path):
    try:
        # Load the image for face recognition
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the pre-trained face detector
        face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            st.error("No faces detected in the image!")
            return None

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            return face
        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Streamlit interface
st.title("Selfie Upload for Attendance System")

# Input fields for name and enrollment number
name = st.text_input("Enter your name:")
enrollment = st.text_input("Enter your enrollment number:")

# Image upload widget
uploaded_image = st.file_uploader("Upload your selfie image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Save the image when the Save button is clicked
    if st.button("Save Image"):
        if name and enrollment:
            file_path = save_uploaded_file(uploaded_image, name, enrollment)
            if file_path:
                st.success(f"Image saved successfully: {file_path}")
            else:
                st.error("Failed to save the image.")
        else:
            st.error("Please provide both name and enrollment number.")

    # Process the image when the Process button is clicked
    if st.button("Process Image"):
        file_path = save_uploaded_file(uploaded_image, name, enrollment)
        if file_path:
            face_image = process_selfie_image(file_path)
            if face_image is not None:
                st.image(face_image, caption="Processed Face", use_container_width=True)
            else:
                st.error("Failed to process the image.")
