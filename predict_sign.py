import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ---------- Load ML model ----------
model = joblib.load("models/sign_model.pkl")

# ---------- Load MediaPipe model ----------
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

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    predicted_letter = "None"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # Extract 63 features
        features = []
        for lm in hand:
            features.extend([lm.x, lm.y, lm.z])

        features = np.array(features).reshape(1, -1)

        # Predict
        predicted_letter = model.predict(features)[0].upper()

        # Draw skeleton
        for s, e in HAND_CONNECTIONS:
            x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
            x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)

        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

    # Display prediction
    cv2.putText(
        frame,
        f"Predicted: {predicted_letter}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 200, 100),
        3
    )

    cv2.imshow("ML Sign Recognition (A-E)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()