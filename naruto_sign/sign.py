import cv2
import numpy as np
import mediapipe as mp
import os
import time


RunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(os.path.dirname(__file__), "models/hand_landmarker.task")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run: python3 setup_models.py")
    exit(1)

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=2,
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
    
    
    frame_timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    draw_clone = False
    if results.hand_landmarks and len(results.hand_landmarks) == 2:
        hand1 = results.hand_landmark[0]
        hand2 = results.hand_landmark[1]

        h1_index_tip = hand1[8]
        h1_index_knuckle = hand1[5]
        h2_index_tip = hand2[8]
        h2_index_knuckle = hand2[5]

        h1_extended = h1_index_tip.y < h1_index_knuckle.y
        h2_extended = h2_index_tip.y < h2_index_knuckle.y

        if h1_extended and h2_extended:
            h1_x, h1_y  = int(h1_index_tip.x * w), int(h1_index_knuckle.h)
            h2_x, h2_y = int(h2_index_tip.x * w), int(h2_index_tip.y * h)

            distance = np.sqrt((h1_x - h2_x)**2 + (h1_y - h2_y)**2)

            if distance < 80:
                draw_clone = True

        for hand_landmarks in results.hand_landmarks:
            for lm in hand_landmarks:
                cx,cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0,255,0), cv2.FILLED)

            if draw_clone:

                cv2.putText(img, "Shadow Clone Jutsu", (50,80), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,165,,255),3)
            
                clone_frame = img.copy()

                alpha = 0.6

                small_live = cv2.resize(img,(w//2,h))
                small_clone = cv2.resize(clone_frame,(w//2,h))

                small_clone[:, :, 0] = cv2.add(small_chakra[:, :, 0], 50)

                final_display = np.hstack(small_live, small_clone)

            else:

                final_display = img
            

            

    cv2.imshow("Naruto Hand Sign", img)
    if cv2.waitKey(10) == 27: # Press ESC to close
        break

cam.release()
cv2.destroyAllWindows()