
# Face Recognition-Based Attendance System

This project is designed to automate the process of taking attendance using facial recognition technology. The system captures faces from a live video feed, matches them against a database of known individuals, and records the attendance automatically.

## Overview

The primary objective of this project is to develop a system that uses facial recognition to identify individuals in real-time and mark their attendance. The system leverages computer vision and machine learning techniques to accurately detect and recognize faces.

## Components

### Face Detection and Recognition

- **Face Detection**: The system uses the `face_recognition` library to detect faces in a video stream.
- **Face Encoding**: Each face is encoded into a numerical format that allows for comparison with stored face encodings.
- **Face Recognition**: The system compares the detected face encodings against known face encodings to identify individuals.

### Attendance Marking

- **CSV Logging**: Once a face is recognized, the system logs the individual's attendance into a CSV file, along with the timestamp.
- **Text-to-Speech Feedback**: The system uses `pyttsx3` to provide audible feedback when a face is recognized and attendance is marked.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Necessary Python libraries:
  - OpenCV (`cv2`)
  - face_recognition
  - NumPy
  - pyttsx3

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. **Install required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add Known Faces:**

   Place images of the individuals whose attendance needs to be tracked in a directory (e.g., `photos/`). Ensure the filenames are indicative of the person’s name.

### Running the System

1. **Execute the script:**

   ```bash
   python main.py
   ```

2. **Provide real-time video feed:**

   The script uses your device's webcam by default. Ensure that your webcam is active and positioned correctly.

3. **View Attendance Logs:**

   The system will generate a CSV file (e.g., `attendance.csv`) that records the attendance along with timestamps.

## File Structure

```
face-recognition-attendance-system/
├── photos/
│   ├── gp.jpg
│   ├── likith.jpg
│   └── arun.jpg
├── main.py
├── requirements.txt
└── README.md
```

### Code Explanation

#### Imports and Setup

The script imports necessary libraries and initializes components like video capture and the text-to-speech engine.

```python
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import pyttsx3

# Initialize the video capture and TTS engine
video_capture = cv2.VideoCapture(0)
engine = pyttsx3.init()
```

#### Loading Known Faces

The script loads and encodes the faces of known individuals, which are then used for real-time recognition.

```python
known_face_encodings = [
    load_face_encoding("photos/gp.jpg"),
    load_face_encoding("photos/likith.jpg"),
    load_face_encoding("photos/arun.jpg")
]
known_face_names = ["Guruprasad G M", "Likith Niravn", "Arunram R"]
```

#### Attendance Marking

When a face is recognized, the system marks attendance and provides feedback via text-to-speech.

```python
students = known_face_names.copy()

def mark_attendance(name):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, current_time])
    engine.say(f"Attendance marked for {name}")
    engine.runAndWait()
```

### Running the Script

To run the attendance system, simply execute:

```bash
python main.py
```

The system will begin capturing video, recognizing faces, and logging attendance.

## Conclusion

This documentation provides a comprehensive guide to setting up and running the Face Recognition-Based Attendance System. By following the steps outlined, you should be able to deploy the system and automate attendance tracking efficiently.

## Result-Screenshots

![1](https://github.com/user-attachments/assets/a361e9bc-42bc-43e8-8b50-796717cbd278)
![2](https://github.com/user-attachments/assets/349dba23-b499-438c-8d21-1643244b54df)
![3](https://github.com/user-attachments/assets/b1c15ee2-3370-43ad-91c8-8939bdfa7b5c)
![4](https://github.com/user-attachments/assets/f70efb98-b3c1-41de-84bb-12b5d35a077e)


## License

This project is licensed under the MIT License.

