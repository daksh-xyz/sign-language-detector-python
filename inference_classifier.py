import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(page_title="Sign Language Detector", layout="centered")

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
expected_features = 84

# UI controls
start = st.button("▶️ Start Camera")
stop = st.button("⏹️ Stop Camera")

frame_window = st.image([])
fps_display = st.empty()

cap = None
prev_time = time.time()

# Main logic
if start and not stop:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access camera.")
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        data_aux = []
        x_all = []
        y_all = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:2]:  # Process up to 2 hands
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_all.extend(x_coords)
                y_all.extend(y_coords)
                min_x = min(x_coords)
                min_y = min(y_coords)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

        # Padding if less than expected features
        if len(data_aux) < expected_features:
            data_aux.extend([0.0] * (expected_features - len(data_aux)))

        if len(data_aux) == expected_features:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            if x_all and y_all:
                x1 = int(min(x_all) * W) - 10
                y1 = int(min(y_all) * H) - 10
                x2 = int(max(x_all) * W) + 10
                y2 = int(max(y_all) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        fps_display.markdown(f"**FPS:** {fps:.2f}")
        frame_window.image(frame, channels='BGR')

        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("Camera stopped.")
