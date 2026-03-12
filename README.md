# Sign Language Recognition System

Real-time Sign Language Recognition using Python, OpenCV, MediaPipe, and Machine Learning.

This project detects hand signs from webcam, converts them to letters, builds words, and speaks the output.

---
## Demo

![Demo] (screenshot.png)

## Features

- Real-time hand tracking using MediaPipe
- Machine learning classification (Scikit-learn)
- A–Y alphabet detection
- Word builder system
- Text-to-speech output
- Confidence score display
- Smooth prediction filtering
- Keyboard controls UI

---

## Controls

S = Speak  
C = Clear  
SPACE = Add space  
BACKSPACE = Delete last letter  
ESC = Quit  

---

## Supported Letters

A – Y supported  
J and Z not supported (motion-based signs)

---

## Project Structure

```
project_1/
│
├── data/
│   └── signs.csv
│
├── models/
│   ├── hand_landmarker.task
│   └── sign_model.pkl
│
├── collect_data.py
├── fix_dataset.py
├── train_model.py
├── predict_sign.py
├── word_system.py
│
├── requirements.txt
|__ screenshot.png
└── README.md
```

---

## Installation

Clone the repository

```
git clone https://github.com/swethakiran/sign-language-recognition.git
cd sign-language-recognition
```

Install requirements

```
pip install -r requirements.txt
```

Run the system

```
python word_system.py
```

---

## Tech Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib
- PyWin32 (speech)

---

## Notes

- Model trained on custom dataset
- J and Z require motion tracking (not included)
- Works with webcam

---

## Author

Swetha Kiran Veernapu