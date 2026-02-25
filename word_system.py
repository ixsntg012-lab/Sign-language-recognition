import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque, Counter
import pyttsx3
import threading
# ---------- Text-to-Speech Setup ----------

speech_lock = threading.Lock()

def speak(text):
    if text.strip() != "":
        threading.Thread(target=_speak_thread, args=(text,)).start()

def _speak_thread(text):
    if speech_lock.locked():
        return # prevent overlapping speech

def _speak_thread(text):
        engine = pyttsx3.init('sapi5') # create fresh engine each time
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

# ---------- Load ML model ----------
model = joblib.load("models/sign_model.pkl")

# ---------- Load MediaPipe ----------
base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ---------- Hand connections ----------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ---------- Smoothing Settings ----------
prediction_buffer = deque(maxlen=12)
confidence_threshold = 0.6
ADD_DELAY = 18

# ---------- Word Builder ----------
current_text = ""
last_added_letter = ""
frame_counter = 0

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    final_prediction = ""

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        features = []
        for lm in hand:
            features.extend([lm.x, lm.y, lm.z])
        features = np.array(features).reshape(1, -1)

        predicted_letter = model.predict(features)[0].upper()
        prediction_buffer.append(predicted_letter)

        counter = Counter(prediction_buffer)
        most_common_letter, count = counter.most_common(1)[0]
        confidence = count / len(prediction_buffer)

        if confidence >= confidence_threshold:
            final_prediction = most_common_letter

        if final_prediction != "":
            frame_counter += 1
            if final_prediction != last_added_letter and frame_counter > ADD_DELAY:
                current_text += final_prediction
                last_added_letter = final_prediction
                frame_counter = 0

        # Draw skeleton
        for s, e in HAND_CONNECTIONS:
            x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
            x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)

        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

    # ---------- UI ----------
    sage_green = (85, 140, 85)

    cv2.putText(frame, f"Current Letter: {final_prediction}", 
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sage_green, 2)

    cv2.putText(frame, f"Text: {current_text}", 
                (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, sage_green, 3)

    cv2.putText(frame, "SPACE: space | C: clear | S: speak | Q: quit", 
                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sage_green, 2)

    cv2.imshow("Sign to Speech System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        current_text += " "

    if key == ord('c'):
        current_text = ""

    if key == ord('s'):
        speak(current_text)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()