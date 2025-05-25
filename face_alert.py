#!/usr/bin/env python3
import cv2
import pickle
import face_recognition
import numpy as np

# (Optional) for desktop notifications:
# from plyer import notification

# Load known faces
with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)  # {"Alice": [enc1, …], ...}

# Tolerance for face matching: lower = stricter (default ~0.6)
TOLERANCE = 0.5

# Start webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Unable to access webcam")

print("🎥  Starting video feed. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for speed (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = small_frame[:, :, ::-1]

    # Detect faces + compute encodings
    face_locs = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, face_locs)

    for (top, right, bottom, left), face_encoding in zip(face_locs, face_encs):
        # Scale back up
        top, right, bottom, left = top*2, right*2, bottom*2, left*2

        # Compare to known
        matches = []
        for name, enc_list in known_faces.items():
            results = face_recognition.compare_faces(enc_list, face_encoding, TOLERANCE)
            if any(results):
                matches.append(name)

        name = matches[0] if matches else "Unknown"

        # Draw box + label
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        # If recognized, pop up desktop notification (optional)
        # if name != "Unknown":
        #     notification.notify(
        #         title="Face Recognized",
        #         message=f"{name} detected",
        #         timeout=2
        #     )

    cv2.imshow("Face Alert", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

