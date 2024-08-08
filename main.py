import face_recognition
import cv2
import numpy as np
import csv 
from datetime import datetime
import pyttsx3


# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Load known faces with multiple samples
def load_face_encoding(image_path, num_samples=5):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image, num_jitters=10)
    if len(encodings) == 0:
        raise ValueError(f"No face found in {image_path}")
    return encodings[0]

known_face_encodings = [
    load_face_encoding("photos/gp.jpg"),
    load_face_encoding("photos/likith.jpg"),
    load_face_encoding("photos/arun.jpg")
]
known_face_names = ["Guruprasad G M", "Likith Niravn", "Arunram R"]

# List of expected students
students = known_face_names.copy()
recognized_faces = set()
unrecognized_faces = set()

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Parameters for face recognition
RECOGNITION_THRESHOLD = 0.5
CONSECUTIVE_FRAMES = 3

# Dictionary to store face recognition history
face_recognition_history = {}

# Open CSV file for writing attendance
with open(f"{current_date}.csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=2)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Face recognition
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] <= RECOGNITION_THRESHOLD:
                name = known_face_names[best_match_index]
                
                # Update recognition history
                if name not in face_recognition_history:
                    face_recognition_history[name] = 1
                else:
                    face_recognition_history[name] += 1
                
                # Check if the face has been consistently recognized
                if face_recognition_history[name] >= CONSECUTIVE_FRAMES:
                    frame_color = (0, 255, 0)  # Green for recognized faces
                    
                    if name in students:
                        students.remove(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        lnwriter.writerow([name, current_time])
                        text = f"{name} Present"
                        if name not in recognized_faces:
                            engine.say(text)
                            engine.runAndWait()
                            recognized_faces.add(name)
                    else:
                        text = f"{name} marked present"
                else:
                    name = "Unknown"
                    frame_color = (0, 0, 255)  # Red for unrecognized faces
                    text = "Face not recognized"
            else:
                name = "Unknown"
                frame_color = (0, 0, 255)  # Red for unrecognized faces
                text = "Face not recognized"
                
                # Reset recognition history for this face
                for key in list(face_recognition_history.keys()):
                    face_recognition_history[key] = 0
            
            if name == "Unknown" and face_encoding.tobytes() not in unrecognized_faces:
                engine.say(text)
                engine.runAndWait()
                unrecognized_faces.add(face_encoding.tobytes())
            
            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, frame_color, 2)
            
            # Display text output
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Clear unrecognized_faces set after some interval
        if len(unrecognized_faces) > 10:
            unrecognized_faces.clear()

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()