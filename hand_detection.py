import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Hand skeleton connections
# -----------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# -----------------------------
# Load model
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    sign = "No Sign"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # -----------------------------
        # Finger state detection
        # -----------------------------
        tips = [8, 12, 16, 20]
        bases = [6, 10, 14, 18]

        fingers_up = []
        for tip, base in zip(tips, bases):
            fingers_up.append(1 if hand[tip].y < hand[base].y else 0)

        thumb_tip = hand[4]
        index_base = hand[6]
        thumb_out = thumb_tip.x < index_base.x  # right hand assumption

        # -----------------------------
        # A–E SIGN LOGIC
        # -----------------------------
        # A
        if fingers_up == [0, 0, 0, 0] and thumb_out:
            sign = "A"

        # B
        elif fingers_up == [1, 1, 1, 1] and not thumb_out:
            sign = "B"

        # D
        elif fingers_up == [1, 0, 0, 0]:
            sign = "D"

        # E
        elif fingers_up == [0, 0, 0, 0] and not thumb_out:
            sign = "E"

        # C (curved fingers)
        elif fingers_up == [1, 1, 1, 1] and thumb_out:
            bent_count = 0
            for tip, base in zip(tips, bases):
                if abs(hand[tip].y - hand[base].y) < 0.05:
                    bent_count += 1
        

            if bent_count >= 3:
                sign = "C"

   

        # -----------------------------
        # Draw skeleton
        # -----------------------------
        for s, e in HAND_CONNECTIONS:
            x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
            x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (200, 0, 200), 2)

        # Draw landmarks
        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

    # -----------------------------
    # Display sign
    # -----------------------------
    cv2.putText(
        frame,
        f"Sign: {sign}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 200, 100),
        2
    )

    cv2.imshow("Static Sign Recognition (A–E)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
