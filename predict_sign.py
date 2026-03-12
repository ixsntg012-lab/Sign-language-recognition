import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/sign_model.pkl"
HAND_MODEL = "models/hand_landmarker.task"

model = joblib.load(MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

pred_buffer = deque(maxlen=7)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        # draw skeleton
        for s,e in HAND_CONNECTIONS:
            x1,y1 = int(hand[s].x*w), int(hand[s].y*h)
            x2,y2 = int(hand[e].x*w), int(hand[e].y*h)
            cv2.line(frame,(x1,y1),(x2,y2),(200,0,200),2)

        for lm in hand:
            x,y = int(lm.x*w), int(lm.y*h)
            cv2.circle(frame,(x,y),3,(255,0,255),-1)

        # -------- normalization --------

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

        X = pts.flatten().reshape(1,-1)

        # -------- prediction --------

        probs = model.predict_proba(X)[0]
        confidence = max(probs)

        pred = model.predict(X)[0]

        if confidence < 0.6:
            pred = "?"

        pred_buffer.append(pred)

        stable = max(set(pred_buffer), key=pred_buffer.count)

        cv2.putText(
            frame,
            f"{stable.upper()}  {confidence:.2f}",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()