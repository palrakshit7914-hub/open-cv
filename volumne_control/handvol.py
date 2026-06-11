import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import vision
import os
import time

model_path = os.path.join(os.path.dirname(__file__), "models/hand_landmarker.task")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run: python3 setup_models.py")
    exit(1)


base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# Open the default webcam (index 0).
mp_drawing = vision.drawing_utils

cap = cv2.VideoCapture(0)

last_action = ""
last_time = 0 
# Prevent repeated key presses when a gesture is held across multiple frames.
cooldown = 1.5
FAST_COOLDOWN = 0.01

def fingers_up(hand_landmarks):
    """Detect which fingers are up based on landmarks"""
    tips = [8, 12, 16, 20]   # Fingertip indices
    pips = [6, 10, 14, 18]   # PIP joint indices

    fingers = []

    for tip, pip in zip(tips, pips):
        # If tip is above pip, finger is up
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    _,img = cap.read()
    
    frame = cv2.flip(img, 1)  # Flip the frame horizontally for a mirror effect
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = hand_landmarker.detect(mp_image)
    gesture = "Waiting..."

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            connections = [
                (1, 2), (2, 3), (3, 4),          # Thumb
                (5, 6), (6, 7), (7, 8),                  # Index
                (9, 10), (10, 11), (11, 12),             # Middle
                (13, 14), (14, 15), (15, 16),            # Ring
                (17, 18), (18, 19), (19, 20),            # Pinky
                (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17)  # Palm
            ]

            for connection in connections:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            fingers = fingers_up(hand_landmarks)
            total_fingers = sum(fingers)

            current_time = time.time()

            if total_fingers == 0:
                gesture = "Mute"
                if last_action != "Mute" and current_time - last_time > 1.5:  # Keep longer cooldown for mute
                    pyautogui.press("volumemute")
                    last_action = "Mute"
                    last_time = current_time
                    
            elif total_fingers == 1:
                gesture = "Volume Down"
                if current_time - last_time > cooldown:
                    pyautogui.press("volumedown")
                    last_action = "Down"
                    last_time = current_time
                    
            elif total_fingers == 2:
                gesture = "Volume Up"
                if current_time - last_time > cooldown:
                    pyautogui.press("volumeup")
                    last_action = "Up"
                    last_time = current_time
                    
            elif total_fingers == 3:
                gesture = "Max Volume"
                if current_time - last_time > FAST_COOLDOWN:
                  
                    pyautogui.press("volumeup", presses=2)
                    last_action = "Max Volume"
                    last_time = current_time
            elif total_fingers == 4:
                gesture = "Min Volume"
                if current_time - last_time > FAST_COOLDOWN:
                    
                    pyautogui.press("volumedown", presses=2)
                    last_action = "Min Volume"
                    last_time = current_time
            else:
                gesture = f"{total_fingers} Fingers Up (No Action)"

    cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Hand Volume Control using Python", frame)
    waitKey = cv2.waitKey(10)
    if waitKey == 27:
        break

cap.release()
cv2.destroyAllWindows()