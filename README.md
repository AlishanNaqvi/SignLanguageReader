# SignLanguageReader

**SignLanguageReader** is a real-time hand gesture recognition app that detects American Sign Language (ASL) signs using a machine learning model. It combines computer vision and intuitive game mechanics to create an engaging, educational experience.

The project offers two modes:
- **Detection Mode**: Recognizes and displays signs live through the webcam.
- **Game Mode**: Challenges users to match random signs with scoring, lives, and animated feedback.

A demonstration of the project can be found inside the `demo/` folder.

---

## Features
- Real-time ASL hand gesture detection.
- Gamified mode with score tracking and lives system.
- Animated panels for score, streaks, and game over screens.
- Gesture history display to track previous signs.
- Local leaderboard tracking.
- Streamlit web app version in progress.

---

## Technologies Used
- **Python**
- **OpenCV** — Video capture and image processing.
- **MediaPipe** — Hand landmark detection.
- **Scikit-learn** — Machine learning classification.
- **NumPy** — Numerical operations.
- **Streamlit** — (For future web app version)

---

## Project Structure
```
SignLanguageReader/
│
├── app.py               # Main application file
├── model.p              # Trained ML model
├── leaderboard.txt      # Stores game high scores
├── demo/                # Project demonstration clips
├── assets/              # Images and visual assets
├── requirements.txt     # List of dependencies
└── README.md            # Project overview
```

---
