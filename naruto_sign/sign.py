import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import os

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

hand_landmarker = vision.HandLandmarker.create_from_options(options)\

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    img = cv2.flip(img,1)
    h,w,_ = img.shape

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand_landmarker.detect_for_video(rgb_frame, int(cam.get(cv2.CAP_PROP_POS_FRAMES)))
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            # Example: Print the coordinates of the index finger tip   
            index_finger_tip = landmarks[8]
            index_knuckle = landmarks[5]
            print(f"Index Finger Tip Coordinates: (x: {index_finger_tip.x}, y: {index_finger_tip.y}, z: {index_finger_tip.z})")
            print(f"Index Knuckle Coordinates: (x: {index_knuckle.x}, y: {index_knuckle.y}, z: {index_knuckle.z})")

            cv2.putText(img, f"Index Tip: ({index_finger_tip.x:.2f}, {index_finger_tip.y:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Index Knuckle: ({index_knuckle.x:.2f}, {index_knuckle.y:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Naruto Hand Sign", img)
    if cv2.waitKey(10) == 27:
        break
cam.release()
cv2.destroyAllWindows() 