import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv("data/signs.csv")

X = data.iloc[:, :-1]   # 63 landmark features
y = data.iloc[:, -1]    # labels (A–E)

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train KNN model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy*100:.2f}%")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "models/sign_model.pkl")
print("Model saved to models/sign_model.pkl")
