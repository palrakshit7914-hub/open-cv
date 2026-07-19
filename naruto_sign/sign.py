import cv2
import numpy as np
import mediapipe as mp
import os
import time

# 1. Running mode setting correct kiya
RunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(os.path.dirname(__file__), "models/hand_landmarker.task")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run: python3 setup_models.py")
    exit(1)

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

# FIX: running_mode=RunningMode.VIDEO ko yahan add kiya
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
cam = cv2.VideoCapture(0)

print("Naruto Tracking Active! Press 'ESC' to exit.")

while True:
    _, img = cam.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    # FIX: Live video streams ke liye monotonic timestamp dena zaroori hai
    frame_timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            
            # FIX FOR ERROR 2: Raw landmarks ko Protobuf format mein convert kiya taaki drawing utils crash na ho
            hand_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                mp.framework.formats.landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
            ])
            
            # Sahi tarike se skeleton lines draw karna
            mp.solutions.drawing_utils.draw_landmarks(
                img, 
                hand_landmarks_proto, 
                mp.solutions.hands.HAND_CONNECTIONS
            )
            
            # FIX FOR ERROR 1: '.landmark' hata kar directly index se points nikale
            index_finger_tip = hand_landmarks[8]
            index_knuckle = hand_landmarks[5]

            middle_finger_tip = hand_landmarks[12]
            middle_knuckle = hand_landmarks[9]
            
            # Screen par tracking details print karna
            cv2.putText(img, f"Index Tip: ({index_finger_tip.x:.2f}, {index_finger_tip.y:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Index Knuckle: ({index_knuckle.x:.2f}, {index_knuckle.y:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Middle Tip: ({middle_finger_tip.x:.2f}, {middle_finger_tip.y:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(img, f"Middle Knuckle: ({middle_knuckle.x:.2f}, {middle_knuckle.y:.2f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
            # Naruto Sign Logic: Index aur Middle finger dono jab upar (straight) hongi
            if index_finger_tip.y < index_knuckle.y and middle_finger_tip.y < middle_knuckle.y:
                cv2.putText(img, "JUTSU DETECTED!", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Naruto Hand Sign", img)
    if cv2.waitKey(10) == 27: # Press ESC to close
        break

cam.release()
cv2.destroyAllWindows()