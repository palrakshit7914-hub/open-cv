import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import pyautogui
import winsound

x1, y1, x2, y2 = 0, 0, 0, 0
model_path = os.path.join(os.path.dirname(__file__), "models/face_landmarker.task")

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run: python3 setup_models.py")
    exit(1)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

face_mesh = vision.FaceLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

while True:
    _,img = camera.read()
    img = cv2.flip(img, 1) 
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img))
    landmark_points = output.face_landmarks
    if landmark_points:
        landmarks = landmark_points[0]
        for id, landmark in enumerate(landmarks):
            x = int(landmark.x * img.shape[1])
            y = int(landmark.y * img.shape[0])
            if id == 43:
                x1, y1 = x, y
            if id == 287:
                x2, y2 = x,y
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        print(distance)

        if distance > 60:
            cv2.imwrite("selfie.jpg", img)
            winsound.Beep(1000, 500)  # Beep sound for feedback
            cv2.waitKey


    cv2.imshow("Auto selfie for smiling faces ", img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

camera.release()
cv2.destroyAllWindows()
