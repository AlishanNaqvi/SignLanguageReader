import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Webcam
cap = cv2.VideoCapture(0)

# MediaPipe Hands setup (for 2 hands)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Labels
labels_dict = {0: 'Hello', 1: 'Yes', 2: 'Thank You'}
history = []

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction = None
    confidence = 0.0
    predicted_character = ''

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_.clear()
            y_.clear()
            data_aux.clear()

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(proba) * 100
            predicted_character = labels_dict[int(prediction[0])]

            # Draw prediction box
            cv2.rectangle(frame, (x1, y1 - 60), (x2, y1), (50, 200, 50), -1)
            cv2.putText(frame, f'{predicted_character} ({confidence:.1f}%)',
                        (x1 + 10, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            # Update gesture history (avoid duplicates)
            if len(history) == 0 or predicted_character != history[-1]:
                history.append(predicted_character)
                if len(history) > 5:
                    history.pop(0)

    # Draw gesture history sidebar
    sidebar_x = W - 220
    cv2.rectangle(frame, (sidebar_x, 10), (W - 10, 10 + 30 * 6), (240, 240, 240), -1)
    cv2.putText(frame, 'History', (sidebar_x + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    for i, gesture in enumerate(reversed(history)):
        cv2.putText(frame, f'{gesture}', (sidebar_x + 10, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Show the frame
    cv2.imshow('Sign Language Detector 🔤', frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
