MODE = 'game'  # or 'detect'

import pickle
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

# MODE OPTIONS: 'game' or 'detect'
MODE = 'game'

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Hello', 1: 'Yes', 2: 'Thank You'}

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
SMALL = 0.7
MED = 1.0
LARGE = 1.3

# Leaderboard file
LEADERBOARD_FILE = 'leaderboard.txt'
if not os.path.exists(LEADERBOARD_FILE):
    with open(LEADERBOARD_FILE, 'w') as f:
        pass

# Game settings
target_sign = None
score = 0
streak = 0
best_streak = 0
round_time = 5
last_match_time = time.time()
popup_timer = 0
popup_text = ''

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

def pick_new_target():
    return random.choice(list(labels_dict.values()))

def update_leaderboard(score):
    try:
        with open(LEADERBOARD_FILE, 'a') as f:
            f.write(f'{score}\n')
    except:
        pass

def get_top_scores(n=3):
    try:
        with open(LEADERBOARD_FILE, 'r') as f:
            scores = sorted([int(line.strip()) for line in f], reverse=True)
            return scores[:n]
    except:
        return []

if MODE == 'game':
    target_sign = pick_new_target()
# Fancy scanner-style hand drawing
def draw_fancy_hand(frame, hand_landmarks, width, height):
    # Define landmark indices for each finger
    finger_indices = {
        'thumb': list(range(0, 5)),
        'index': list(range(5, 9 + 1)),
        'middle': list(range(9, 13 + 1)),
        'ring': list(range(13, 17 + 1)),
        'pinky': list(range(17, 20 + 1))
    }

    finger_colors = {
        'thumb': (255, 0, 0),      # Blue
        'index': (0, 255, 0),      # Green
        'middle': (0, 0, 255),     # Red
        'ring': (255, 255, 0),     # Yellow
        'pinky': (255, 0, 255)     # Magenta
    }

    landmark_points = []

    for i, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * width), int(lm.y * height)
        landmark_points.append((cx, cy))

    # Draw finger points
    for finger, indices in finger_indices.items():
        color = finger_colors[finger]
        for idx in indices:
            cx, cy = landmark_points[idx]
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 2)  # outer ring
            cv2.circle(frame, (cx, cy), 4, color, -1)           # inner dot

    # Draw lines (custom)
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        x1, y1 = landmark_points[start_idx]
        x2, y2 = landmark_points[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux, x_, y_ = [], [], []
    predicted_character = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_fancy_hand(frame, hand_landmarks, W, H)

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

    # DISPLAY MODES
    if MODE == 'detect':
        cv2.putText(frame, f'Detected: {predicted_character} ({confidence:.1f}%)',
                    (10, 30), FONT, MED, (50, 200, 50), 3)
    else:
        # GAME MODE UI
        cv2.putText(frame, f'Score: {score}', (10, 40), FONT, MED, (0, 255, 0), 3)
        cv2.putText(frame, f'Target: {target_sign}', (10, 80), FONT, LARGE, (255, 0, 0), 3)
        if predicted_character:
            cv2.putText(frame, f'You: {predicted_character}', (10, 120), FONT, MED, (100, 100, 100), 3)

        # Streak
        cv2.putText(frame, f'Streak: {streak}', (10, 160), FONT, SMALL, (255, 165, 0), 3)
        cv2.putText(frame, f'Best Streak: {best_streak}', (10, 190), FONT, SMALL, (255, 165, 0), 3)

        # Leaderboard
        top_scores = get_top_scores()
        cv2.putText(frame, 'Top Scores:', (W - 200, 40), FONT, SMALL, (200, 200, 200), 3)
        for i, s in enumerate(top_scores):
            cv2.putText(frame, f'{i+1}. {s}', (W - 200, 70 + i*30), FONT, SMALL, (180, 180, 180), 3)

        # Timer
        time_left = round_time - (time.time() - last_match_time)
        if time_left <= 0:
            target_sign = pick_new_target()
            streak = 0
            last_match_time = time.time()
        else:
            cv2.putText(frame, f'Time Left: {int(time_left)}s', (10, 230), FONT, SMALL, (0, 0, 255), 3)

        # Match success
        if predicted_character == target_sign:
            score += 1
            streak += 1
            best_streak = max(best_streak, streak)
            popup_text = random.choice(["Correct!", "Nice!", "Well done!"])
            popup_timer = 15
            target_sign = pick_new_target()
            last_match_time = time.time()

    # Subtle popup text (if any)
    if popup_timer > 0:
        cv2.putText(frame, popup_text, (W//2 - 100, H//2),
                    FONT, 1.5, (0, 200, 0), 3)
        popup_timer -= 1

    cv2.imshow("Sign Language Game 🎯", frame)

    key = cv2.waitKey(1)
    if key == 27:
        if MODE == 'game':
            update_leaderboard(score)
        break

cap.release()
cv2.destroyAllWindows()
