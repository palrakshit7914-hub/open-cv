import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import os



model_path = os.path.join(os.path.dirname(__file__), "models/face_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

face_mesh = vision.FaceLandmarker.create_from_options(options)
cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB))
    cv2.imshow("Eye Control Mouse", img)
    key = cv2.waitKey(100)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()