import tkinter as tk
from tkinter import messagebox
import cv2
import csv
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import face_recognition  # Import face_recognition library

# Global configurations
HAARCASCADE_PATH = "haarcascades/haarcascade_frontalface_default.xml"
TRAINING_IMAGE_PATH = "TrainingImage/"
TRAINING_LABEL_PATH = "TrainingImageLabel/"
STUDENT_DETAILS_PATH = "StudentDetails/student_data.csv"
ATTENDANCE_LOG_PATH = "Attendance/Auto_Attendance_Logs/attendance_log.csv"

# Main Window setup
window = tk.Tk()
window.title("Attendance Management System")
window.geometry('1280x720')
window.configure(background='lightgrey')

# Helper function to show error messages
def show_error(message):
    messagebox.showerror("Error", message)

# Helper function to show success messages
def show_success(message):
    messagebox.showinfo("Success", message)

# Function to enhance the image
def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(gray)
    return enhanced_img

# Extract face features using face_recognition
def extract_features(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    return encodings[0] if encodings else None

# Function to load the trained model
def load_trained_model():
    model_path = f"{TRAINING_LABEL_PATH}/Trainner.yml"
    
    # Check if the model file exists before proceeding
    if not os.path.exists(model_path):
        show_error("Model not found. Please train the model first!")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    return recognizer

# Function for automatic attendance with retry on timeout
def automatic_attendance():
    recognizer = load_trained_model()
    if not recognizer:
        return  # Exit if the model is not found
    
    cam = cv2.VideoCapture(0)
    df = pd.read_csv(STUDENT_DETAILS_PATH)
    col_names = ['Enrollment', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    time_limit = 10  # seconds to wait for face recognition
    start_time = time.time()
    recognized = False  # Flag to check if the person has been recognized

    known_face_encodings = []  # Store known face encodings

    # Load and encode known faces from the training set
    for imagePath in os.listdir(TRAINING_IMAGE_PATH):
        img = Image.open(os.path.join(TRAINING_IMAGE_PATH, imagePath))
        img_np = np.array(img)
        face_encoding = extract_features(img_np)
        if face_encoding is not None:
            known_face_encodings.append(face_encoding)

    while True:
        ret, img = cam.read()
        if not ret:
            continue  # Skip if frame is not captured correctly

        # Convert to RGB for face_recognition (this is necessary since face_recognition requires RGB format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get face locations
        face_locations = face_recognition.face_locations(img_rgb)

        for (top, right, bottom, left) in face_locations:
            # Enhance image and apply multiple recognition patterns
            face_encoding = face_recognition.face_encodings(img_rgb, [(top, right, bottom, left)])

            if face_encoding:  # Only proceed if we have a face encoding
                face_encoding = face_encoding[0]

                # Compare with known faces in the model
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                if True in matches:
                    match_index = matches.index(True)
                    name = df.iloc[match_index]['Name']
                    enrollment = df.iloc[match_index]['Enrollment']
                    date = datetime.datetime.now().strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.now().strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = [enrollment, name, date, timeStamp]

                    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
                    cv2.putText(img, f"{name} - {enrollment}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Save attendance when a match is found
                    attendance.to_csv(ATTENDANCE_LOG_PATH, index=False, mode='a', header=not os.path.exists(ATTENDANCE_LOG_PATH))

                    # Show success message once attendance is logged
                    show_success(f"Attendance logged for {name} - {enrollment}")

                    recognized = True  # Set recognized flag to True

        # Display the frame with bounding boxes and name
        cv2.imshow("Attendance", img)

        # Check if time has passed and retry if no face was recognized
        if time.time() - start_time > time_limit and not recognized:
            show_error("Face recognition timeout! Retrying...")
            start_time = time.time()  # Reset the timer and retry

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Remove duplicates from the attendance dataframe
    attendance = attendance.drop_duplicates(['Enrollment'], keep='first')
    cam.release()
    cv2.destroyAllWindows()

    # Save the attendance log to CSV again after loop ends
    attendance.to_csv(ATTENDANCE_LOG_PATH, index=False, mode='a', header=not os.path.exists(ATTENDANCE_LOG_PATH))
    show_success("Attendance logged successfully!")

# GUI components
enrollment_label = tk.Label(window, text="Enter Enrollment: ", bg="lightgrey", font=('times', 15))
enrollment_label.place(x=200, y=200)
enrollment_input = tk.Entry(window, width=20, font=('times', 15))
enrollment_input.place(x=400, y=200)

name_label = tk.Label(window, text="Enter Name: ", bg="lightgrey", font=('times', 15))
name_label.place(x=200, y=250)
name_input = tk.Entry(window, width=20, font=('times', 15))
name_input.place(x=400, y=250)

auto_att_btn = tk.Button(window, text="Automatic Attendance", command=automatic_attendance, bg="blue", fg="white", font=('times', 15))
auto_att_btn.place(x=600, y=300)

quit_btn = tk.Button(window, text="Quit", command=window.destroy, bg="red", fg="white", font=('times', 15))
quit_btn.place(x=800, y=300)

window.mainloop()
