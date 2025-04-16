import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import mediapipe as mp
import pickle
import av

st.set_page_config(page_title="Sign Language Detector", layout="centered")
st.title("🤟 Live Sign Language Detector (Two‑Hand Support)")

# --- 1. Load your trained model once ---
model = pickle.load(open('model.p', 'rb'))['model']
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
expected_features = 84  # 42 features per hand × 2 hands

# --- 2. Initialize MediaPipe once ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,              # continuous stream
    max_num_hands=2,                      # up to 2 hands
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

class SignLangProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        This method is called for each video frame; it must return a new frame.
        """
        img = frame.to_ndarray(format="bgr24")  # Convert to NumPy array (H×W×3)
        H, W, _ = img.shape

        # Process with MediaPipe
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        data_aux = []
        x_all = []
        y_all = []

        # Draw landmarks & collect features for up to 2 hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # extract x,y coords
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_all.extend(xs)
                y_all.extend(ys)
                min_x, min_y = min(xs), min(ys)

                # normalize & flatten
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

        # pad if only one (or zero) hand detected
        if len(data_aux) < expected_features:
            data_aux.extend([0.0] * (expected_features - len(data_aux)))

        # Predict if vector is correct length
        if len(data_aux) == expected_features:
            pred = model.predict([np.asarray(data_aux)])[0]
            char = labels_dict[int(pred)]

            # draw bounding box + label
            if x_all and y_all:
                x1 = int(min(x_all) * W) - 10
                y1 = int(min(y_all) * H) - 10
                x2 = int(max(x_all) * W) + 10
                y2 = int(max(y_all) * H) + 10

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    img, char, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA
                )

        # Return the annotated frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 3. Launch the webrtc component ---
webrtc_streamer(
    key="sign-language-detector",
    video_processor_factory=SignLangProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

