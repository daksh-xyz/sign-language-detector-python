import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Define the feature vector length for one hand
features_per_hand = 42

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        hand_features = []  # To store features from detected hands
        
        if results.multi_hand_landmarks:
            # Process at most two hands
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # Get x and y coordinates for all landmarks
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                min_x = min(x_coords)
                min_y = min(y_coords)
                features = []
                # Normalize coordinates by subtracting the minimum values
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - min_x)
                    features.append(lm.y - min_y)
                hand_features.extend(features)
        
        # If only one hand is detected, pad with zeros for the second hand
        if len(hand_features) < 2 * features_per_hand:
            hand_features.extend([0.0] * (2 * features_per_hand - len(hand_features)))
        
        # You can decide to only add samples with two hands by uncommenting the next lines:
        # if len(results.multi_hand_landmarks) < 2:
        #     continue
        
        data.append(hand_features)
        labels.append(dir_)

# Save the dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
