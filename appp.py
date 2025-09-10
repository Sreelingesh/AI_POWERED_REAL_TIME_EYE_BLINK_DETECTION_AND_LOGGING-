import cv2
import mediapipe as mp
import pandas as pd
import os
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# CSV log file path
log_file = "logs.csv"

# If file doesn't exist, create it with headers
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["Timestamp", "Event"])
    df.to_csv(log_file, index=False)

# Open webcam
cap = cv2.VideoCapture(0)

def log_event(event):
    """Log an event with timestamp to CSV"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, event]], columns=["Timestamp", "Event"])
    df.to_csv(log_file, mode='a', header=False, index=False)
    print(f"Logged: {event} at {timestamp}")

# Eye aspect ratio calculation (using landmarks)
def get_eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    points = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    # Vertical distances
    v1 = abs(points[1][1] - points[5][1])
    v2 = abs(points[2][1] - points[4][1])
    # Horizontal distance
    h = abs(points[0][0] - points[3][0])
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold
EYE_CLOSED_THRESH = 0.2
eye_closed_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            # Calculate EAR for both eyes
            left_ear = get_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = get_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            
            if ear < EYE_CLOSED_THRESH:
                eye_closed_frames += 1
                cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                # Log once every 15 frames (~0.5 sec at 30fps)
                if eye_closed_frames % 15 == 0:
                    log_event("Eyes Closed")
            else:
                eye_closed_frames = 0
                cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    # Show window
    cv2.imshow("Eye Logger", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
