import cv2
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- Setup paths ----------
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "signs.csv")
MODEL_PATH = "models/hand_landmarker.task"

os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Letters (skip J and Z) ----------
LETTERS = [l for l in "abcdefghijklmnopqrstuvwxyz" if l not in ["j","z"]]

# ---------- Load MediaPipe model ----------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# ---------- Hand skeleton connections ----------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ---------- Create CSV header ----------
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

# ---------- Load counts ----------
counts = {l:0 for l in LETTERS}

if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label in counts:
                counts[label] += 1

# ---------- Camera ----------
cap = cv2.VideoCapture(0)

print("Instructions:")
print("Press letters a-z to collect samples")
print("J and Z skipped (dynamic gestures)")
print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    #Draw landmarks
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        for s, e in HAND_CONNECTIONS:
            x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
            x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (200,0,200), 2)

        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x,y), 3, (255,0,255), -1)

    #Counter text
    letters1 = LETTERS[:12]
    letters2 = LETTERS[12:]

    line1 = " ".join([f"{l.upper()}:{counts[l]}" for l in letters1])
    line2 = " ".join([f"{l.upper()}:{counts[l]}" for l in letters2])

    sage_green = (85,140,85)

    cv2.putText(frame, line1, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sage_green, 2)
    cv2.putText(frame, line2, (20,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sage_green, 2)

    cv2.putText(
        frame,
        "Press A-Z to save | J,Z skipped | Q to quit",
        (20,85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        sage_green,
        2
    )

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    #Save sample
    if result.hand_landmarks and ord('a') <= key <= ord('z'):
        letter = chr(key)

        if letter in LETTERS:

            hand = result.hand_landmarks[0]
            row = []

            for lm in hand:
                row.extend([lm.x, lm.y, lm.z])

            row.append(letter)

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            counts[letter] += 1

            print(f"Saved sample for: {letter.upper()}")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()