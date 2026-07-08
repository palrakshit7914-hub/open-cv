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