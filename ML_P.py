import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time

# -----------------------------
# 1. LOAD SVM MODEL (MUST be trained with probability=True)
# -----------------------------
svm_model = pickle.load(open("svm_mediapipe.pkl_1", "rb"))
print(" SVM Model Loaded")

# -----------------------------
# 2. INITIALIZE MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# 3. TEXT-TO-SPEECH ENGINE
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

last_spoken = ""
last_time = 0


# -----------------------------
# FUNCTION: Extract 63 Landmark Features
# -----------------------------
def extract_landmarks_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand_landmarks = result.multi_hand_landmarks[0]

    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features).reshape(1, -1), frame


# -----------------------------
# 4. START WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

print("\n🎥 Webcam started... Show ASL signs to the camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features, annotated_frame = extract_landmarks_from_frame(frame)

    if features is not None:
        # SVM Prediction
        pred = svm_model.predict(features)[0]

        # Confidence (requires probability=True during training)
        proba = svm_model.predict_proba(features)[0]
        max_conf = np.max(proba)
        conf_percent = round(max_conf * 100, 2)

        # Display prediction + confidence
        cv2.putText(
            annotated_frame,
            f"{pred}  ({conf_percent}%)",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # Speak only when new letter OR after delay
        current_time = time.time()
        if pred != last_spoken and conf_percent > 70:
            engine.say(pred)
            engine.runAndWait()
            last_spoken = pred
            last_time = current_time

    else:
        cv2.putText(
            frame,
            "No hand detected",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        annotated_frame = frame

    cv2.imshow("ASL Detection - Mediapipe + SVM + Confidence", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
