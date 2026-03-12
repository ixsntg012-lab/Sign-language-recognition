import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

CSV_PATH = "data/signs.csv"
MODEL_PATH = "models/sign_model.pkl"

df = pd.read_csv(CSV_PATH)

X_raw = df.drop("label", axis=1).values
y = df["label"].values

X = []

for row in X_raw:

    wrist_x = row[0]
    wrist_y = row[1]
    wrist_z = row[2]

    pts = []

    for i in range(0,63,3):
        x = row[i] - wrist_x
        yv = row[i+1] - wrist_y
        z = row[i+2] - wrist_z
        pts.append([x,yv,z])

    pts = np.array(pts)

    # scale normalization
    max_dist = np.max(np.linalg.norm(pts, axis=1))

    pts = pts / max_dist

    X.append(pts.flatten())

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training:", len(X_train))
print("Testing:", len(X_test))

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
    )

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

joblib.dump(model, MODEL_PATH)

print("Saved")