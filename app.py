import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Hello', 1: 'Yes', 2: 'Thank You'}

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# UI
st.title("🤟 Live Sign Language Detection")
FRAME_WINDOW = st.image([])

history = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux, x_, y_ = [], [], []

    predicted_character = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        if data_aux:
            prediction = model.predict([np.asarray(data_aux)])
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(proba) * 100
            predicted_character = labels_dict[int(prediction[0])]

            if len(history) == 0 or predicted_character != history[-1]:
                history.append(predicted_character)
                if len(history) > 5:
                    history.pop(0)

            cv2.putText(frame, f'{predicted_character} ({confidence:.1f}%)',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)

    # Show history
    for i, h in enumerate(reversed(history)):
        cv2.putText(frame, f'{i+1}. {h}', (10, 60 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    FRAME_WINDOW.image(frame, channels="BGR")
