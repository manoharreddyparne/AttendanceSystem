import tkinter as tk
from tkinter import messagebox
import cv2
import csv
import os
import numpy as np
from PIL import ImageTk, Image
import pandas as pd
import datetime
import time
import face_recognition  # Import face_recognition library

# Global configurations
HAARCASCADE_PATH = "haarcascades/haarcascade_frontalface_default.xml"
TRAINING_IMAGE_PATH = "TrainingImage/"
TRAINING_LABEL_PATH = "TrainingImageLabel/"
STUDENT_DETAILS_PATH = "StudentDetails/student_data.csv"
ATTENDANCE_LOG_PATH = "Attendance/Auto_Attendance_Logs/"

# Main Window setup
window = tk.Tk()
window.title("Attendance Management System")
window.geometry('1280x720')
window.configure(background='lightgrey')

# Define log_label globally so it's accessible across functions
log_label = tk.Label(window, text="", bg="lightgrey", fg="green", font=('times', 12, 'bold'))
log_label.place(x=800, y=275)  # Adjust the position as needed

# Global variables
log_shown = False
attendance_logged = False
failed_to_recognize = False  # Define this variable here globally
retry_count = 0

# Helper function to show error messages
def show_error(message):
    messagebox.showerror("Error", message)

# Helper function to show success messages
def show_success(message):
    messagebox.showinfo("Success", message)

# Function to preprocess and enhance the image
def preprocess_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram equalization to improve the contrast (lighting)
    gray = cv2.equalizeHist(gray)

    # Additional enhancement like sharpening (optional)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    return sharpened

# Function for Data Augmentation (for better training)
def augment_image(image):
    # Flip the image horizontally
    flipped = cv2.flip(image, 1)

    # Rotate the image by a random degree
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    angle = np.random.randint(-30, 30)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Adjust brightness (change the intensity)
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=50)  # Increase brightness

    return [flipped, rotated, bright]

# Function to check if the enrollment already exists in the student data CSV
def check_duplicate_registration(enrollment):
    if not os.path.exists(STUDENT_DETAILS_PATH):
        return False
    df = pd.read_csv(STUDENT_DETAILS_PATH)
    if enrollment in df['Enrollment'].values:
        return True
    return False

# Function to capture and save training images
def take_images():
    enrollment = enrollment_input.get()
    name = name_input.get()
    if enrollment == '' or name == '':
        show_error("Please enter Enrollment and Name!")
        return

    # Check for numeric enrollment and alphabetic name
    if not enrollment.isdigit():
        show_error("Enrollment must be a number!")
        return
    if not name.isalpha():
        show_error("Name must only contain alphabets!")
        return

    # Check for duplicate registration
    if check_duplicate_registration(enrollment):
        show_error(f"Enrollment {enrollment} is already registered!")
        return

    cam = cv2.VideoCapture(0)
    sampleNum = 0
    os.makedirs(TRAINING_IMAGE_PATH, exist_ok=True)

    while True:
        ret, img = cam.read()
        processed_img = preprocess_image(img)  # Apply preprocessing

        # Use face_recognition to detect faces
        face_locations = face_recognition.face_locations(processed_img)

        # Save only up to 5 images per student
        if sampleNum >= 5:
            break

        for (top, right, bottom, left) in face_locations:
            sampleNum += 1
            file_name = f"{TRAINING_IMAGE_PATH}/{enrollment}_{name}_{sampleNum}.jpg"
            
            # Check if the image already exists
            if os.path.exists(file_name):
                show_error(f"Image already exists for Enrollment: {enrollment} and Name: {name}")
                return

            cv2.imwrite(file_name, processed_img[top:bottom, left:right])  # Save the processed image
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.imshow('Capturing Images', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(STUDENT_DETAILS_PATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([enrollment, name])
    
    show_success(f"Images saved for Enrollment: {enrollment}, Name: {name}")

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Using LBPH model for training
    imagePaths = [os.path.join(TRAINING_IMAGE_PATH, f) for f in os.listdir(TRAINING_IMAGE_PATH)]
    
    face_samples = []
    ids = []

    for imagePath in imagePaths:
        try:
            img = Image.open(imagePath).convert('L')  # Convert image to grayscale
            imageNp = np.array(img, 'uint8')  # Convert to NumPy array
            
            filename = os.path.basename(imagePath)
            enrollment_id = int(filename.split("_")[0])  # Extract enrollment ID

            face_locations = face_recognition.face_locations(imageNp)

            if len(face_locations) == 0:
                print(f"Warning: No face detected in {filename}. Skipping...")
                continue

            for (top, right, bottom, left) in face_locations:
                face_samples.append(imageNp[top:bottom, left:right])  # Store grayscale face
                ids.append(enrollment_id)

        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    if len(face_samples) == 0:
        show_error("No faces found in training images. Please ensure that the images are correctly placed and contain faces.")
        return

    recognizer.train(face_samples, np.array(ids))  # Train the model
    os.makedirs(TRAINING_LABEL_PATH, exist_ok=True)
    model_path = f"{TRAINING_LABEL_PATH}/Trainner.yml"
    recognizer.save(model_path)  # Save the trained model

    show_success(f"Model trained successfully! The model has been saved to {model_path}.")

# Define global variables for video label, close button, and webcam frame
video_label = None
close_btn = None
webcam_frame = None

# Initialize cam as None
cam = None
def close_webcam():
    global cam, video_label, close_btn, webcam_frame, running

    # Stop the webcam feed and release the camera only if it was opened
    if cam is not None and cam.isOpened():
        cam.release()

    cv2.destroyAllWindows()

    # Destroy the webcam and other UI elements
    if video_label:
        video_label.destroy()
    if webcam_frame:
        webcam_frame.destroy()
    if close_btn:
        close_btn.destroy()

    # Reset the running flag to stop processing frames
    running = False

    # Re-enable the UI components (buttons and inputs)
    enrollment_input.config(state='normal')
    name_input.config(state='normal')
    take_img_btn.config(state='normal')
    train_img_btn.config(state='normal')
    auto_att_btn.config(state='normal')
    quit_btn.config(state='normal')

    # Clear the log message
    log_label.config(text="")

# Extract face features using face_recognition
def extract_features(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    return encodings[0] if encodings else None

# Function to save student data with serial number to student_data.csv after training
def save_student_data(enrollment, name):
    try:
        # Read existing data from student_data.csv
        if os.path.exists("StudentDetails/student_data.csv"):
            df = pd.read_csv("StudentDetails/student_data.csv")
            
            # Check if the record already exists in the CSV
            if enrollment in df['Enrollment'].values:
                show_error(f"Enrollment {enrollment} is already registered!")
                return False  # Return False if the record exists

            # Create a new serial number based on the highest existing serial number
            max_serial = df['Serial Number'].max() if not df.empty else 0
            new_serial = max_serial + 1
        else:
            # If the file doesn't exist, start with serial number 1
            new_serial = 1

        # Append new student data with serial number
        with open("StudentDetails/student_data.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([new_serial, enrollment, name])
        return True  # Return True after successfully saving the data
    except Exception as e:
        show_error(f"Error saving student data: {str(e)}")
        return False

# Function to get today's date in the desired format (e.g., "10_12_2024")
def get_today_date():
    return datetime.datetime.now().strftime("%d_%m_%Y")

# Function to get today's attendance file path (based on the date)
def get_attendance_file_path():
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    file_name = f"attendance_{today_date}.csv"
    file_path = os.path.join(ATTENDANCE_LOG_PATH, file_name)
    return file_path
# Function to check if the student has already marked attendance today
def is_attendance_marked_today(enrollment, today_date):
    file_path = get_attendance_file_path()
    
    # Check if the file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Check if the student has marked attendance for the given date
        if enrollment in df['Enrollment'].values and today_date in df['Date'].values:
            return True  # Attendance already marked for this student today
    return False

# Function to log attendance
def log_attendance(enrollment, name, today_date, present=True):
    global log_shown
    global attendance_logged

    # Get today's attendance file path
    file_path = get_attendance_file_path()

    # Check if the attendance file exists, if not, create it with headers
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Status'])

    # Check if the student has already been marked present for today
    if any((df['Enrollment'] == enrollment) & (df['Date'] == today_date)):
        if not log_shown:  # Only show the message once after 3 seconds if already marked
            log_label.config(text="Attendance already marked.")
            window.after(3000, lambda: log_label.config(text=""))  # Clear after 3 seconds
            log_shown = True
        attendance_logged = True
        return "Attendance already marked."

    # Create a new row to add
    status = 'Present' if present else 'Absent'
    new_row = pd.DataFrame([{'Enrollment': enrollment, 'Name': name, 'Date': today_date, 'Status': status}])

    # Add the new row to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame to the file
    df.to_csv(file_path, index=False)

    # Display thank you message for 4 seconds
    log_label.config(text=f"Attendance marked for {name}. Thank you.")
    window.after(4000, lambda: log_label.config(text=""))  # Clear after 4 seconds

    attendance_logged = True
    return f"Attendance marked for {name} on {today_date}."

# Function to handle retry logic if recognition fails
# Function to handle retry logic if recognition fails
def handle_failed_recognition():
    global retry_count, failed_to_recognize

    if retry_count >= 3:
        log_label.config(text="Failed to recognize. Retrying...")
        window.after(2000, lambda: log_label.config(text=""))  # Clear after 2 seconds
        retry_count = 0
    else:
        retry_count += 1
        failed_to_recognize = True

def automatic_attendance():
    global cam, running, log_shown, video_label, close_btn, webcam_frame

    # Disable the buttons and inputs while processing
    enrollment_input.config(state='disabled')
    name_input.config(state='disabled')
    take_img_btn.config(state='disabled')
    train_img_btn.config(state='disabled')
    auto_att_btn.config(state='disabled')
    quit_btn.config(state='disabled')

    # Create a new frame for the webcam feed
    webcam_frame = tk.Frame(window, bg="black")
    webcam_frame.place(x=630, y=25, width=400, height=250)

    # Add a label for the webcam feed
    video_label = tk.Label(webcam_frame, bg="black")
    video_label.pack(expand=True, fill="both")

    # Add a "Close" button for stopping the webcam
    close_btn = tk.Button(window, text="Close", command=lambda: close_webcam(), bg="red", fg="white", font=('times', 15))
    close_btn.place(x=900, y=300)

    # Log label for attendance messages
    log_label = tk.Label(window, text="", bg="lightgrey", fg="green", font=('times', 12, 'bold'))
    log_label.place(x=800, y=275)

    # Initialize webcam
    model_path = f"{TRAINING_LABEL_PATH}/Trainner.yml"

    # Check if the model file exists
    if not os.path.exists(model_path):
        if len(os.listdir(TRAINING_IMAGE_PATH)) == 0:
            show_error("No training images found! Please add some training images for face recognition.")
            close_webcam()  # Close the webcam and return to main page
            return
        show_error("Model not found. Please train the model first!")
        close_webcam()  # Close the webcam and return to main page
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
    except cv2.error as e:
        show_error(f"Error loading model: {e}")
        close_webcam()  # Close the webcam and return to main page
        return

    # Initialize the webcam feed only if everything is valid
    cam = cv2.VideoCapture(0)  # Open webcam

    if not cam.isOpened():
        show_error("Unable to access the webcam.")
        close_webcam()  # Close the webcam and return to main page
        return

    # Load student details and known face encodings
    df = pd.read_csv(STUDENT_DETAILS_PATH)
    known_face_encodings = []

    for imagePath in os.listdir(TRAINING_IMAGE_PATH):
        img = Image.open(os.path.join(TRAINING_IMAGE_PATH, imagePath))
        img_np = np.array(img)
        face_encoding = extract_features(img_np)
        if face_encoding is not None:
            known_face_encodings.append(face_encoding)

    # Process webcam feed and recognize faces
    def process_frame():
        global running, log_shown

        if not cam.isOpened():
            return  # Skip if camera is not opened correctly

        ret, img = cam.read()
        if not ret:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        attendance_logged = False
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')

        for (top, right, bottom, left) in face_locations:
            face_encoding = face_recognition.face_encodings(img_rgb, [(top, right, bottom, left)])
            if face_encoding:
                face_encoding = face_encoding[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    name = df.iloc[match_index]['Name']
                    enrollment = df.iloc[match_index]['Enrollment']

                    # Log attendance and mark as present
                    status = log_attendance(enrollment, name, today_date, present=True)
                    log_label.config(text=status)
                    attendance_logged = True

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)

        if cam.isOpened():
            video_label.after(10, process_frame)

    # Start processing frames
    process_frame()

# GUI components
enrollment_label = tk.Label(window, text="Enter Enrollment: ", bg="lightgrey", font=('times', 15))
enrollment_label.place(x=200, y=200)
enrollment_input = tk.Entry(window, width=20, font=('times', 15))
enrollment_input.place(x=400, y=200)

name_label = tk.Label(window, text="Enter Name: ", bg="lightgrey", font=('times', 15))
name_label.place(x=200, y=250)
name_input = tk.Entry(window, width=20, font=('times', 15))
name_input.place(x=400, y=250)

take_img_btn = tk.Button(window, text="Take Images", command=take_images, bg="blue", fg="white", font=('times', 15))
take_img_btn.place(x=200, y=300)

train_img_btn = tk.Button(window, text="Train Model", command=train_images, bg="blue", fg="white", font=('times', 15))
train_img_btn.place(x=400, y=300)

auto_att_btn = tk.Button(window, text="Automatic Attendance", command=automatic_attendance, bg="blue", fg="white", font=('times', 15))
auto_att_btn.place(x=600, y=300)

quit_btn = tk.Button(window, text="Quit", command=window.destroy, bg="red", fg="white", font=('times', 15))
quit_btn.place(x=800, y=300)

window.mainloop()
