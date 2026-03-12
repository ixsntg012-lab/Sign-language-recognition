import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import win32com.client

# ---------------- SPEECH ----------------

speaker = win32com.client.Dispatch("SAPI.SpVoice")

def speak(text):
    speaker.Speak(text)


# ---------------- PATHS ----------------

MODEL_PATH = "models/sign_model.pkl"
HAND_MODEL = "models/hand_landmarker.task"

model = joblib.load(MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ---------------- VARIABLES ----------------

pred_buffer = deque(maxlen=7)

current_text = ""

last_stable = ""
stable_start = time.time()

ADD_DELAY = 1.2

cap = cv2.VideoCapture(0)


# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()
    if not ret:
        break


    # -------- DETECTION --------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)


    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        wrist = hand[0]

        pts = []

        for lm in hand:
            pts.append([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        pts = np.array(pts)

        max_dist = np.max(np.linalg.norm(pts, axis=1))
        pts = pts / max_dist

        X = pts.flatten().reshape(1, -1)

        probs = model.predict_proba(X)[0]
        confidence = max(probs)

        conf_percent = int(confidence * 100)

        pred = model.predict(X)[0]

        if confidence < 0.6:
            pred = "?"

        pred_buffer.append(pred)

        stable = max(set(pred_buffer), key=pred_buffer.count)

        if stable != last_stable:
            last_stable = stable
            stable_start = time.time()

        else:
            if stable != "?" and time.time() - stable_start > ADD_DELAY:

                current_text += stable

                stable_start = time.time()
                pred_buffer.clear()
                last_stable = ""

    else:
        pred_buffer.clear()
        last_stable = ""


    # -------- DISPLAY --------

    # TITLE
    cv2.putText(
        frame,
        "Sign Language Recognition System",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 0, 0),
        2
    )

    # WORD
    cv2.putText(
        frame,
        f"Word: {current_text}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    # CONFIDENCE
    if result.hand_landmarks:
        cv2.putText(
            frame,
            f"Conf: {conf_percent}%",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # CONTROLS BOTTOM
    h, w, _ = frame.shape

    cv2.putText(
        frame,
        "S=Speak  C=Clear  SPACE=Space  BACK=Delete  ESC=Quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (50, 50, 50),
        1
    )

    cv2.imshow("Sign Language Recognition System", frame)


    # -------- KEYS --------

    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == ord('c'):
        current_text = ""

    if key == ord('s') and current_text != "":
        speak(current_text)

    if key == 32:
        current_text += " "

    if key == 8:
        current_text = current_text[:-1]


cap.release()
cv2.destroyAllWindows()