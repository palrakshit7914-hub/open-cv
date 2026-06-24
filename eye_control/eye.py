import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import os
import screeninfo

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

screen = screeninfo.get_monitors()[0]
screen_w, screen_h = screen.width, screen.height

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

smooth_x, smooth_y = 0, 0
damping = 0.08

while True:
    _,img = cam.read()
    img = cv2.flip(img, 1)
    h,w,_ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed = face_mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB))
    all_faces = processed.face_landmarks
    
    if all_faces:
        landmarks_points = all_faces[0]
        for i, landmark in enumerate(landmarks_points[474:478]):
            y = int(landmark.y * h)
            x = int(landmark.x * w)
            cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)
            

        left_eye = [landmarks_points[145], landmarks_points[159]]
        for landmark in left_eye:
            y = int(landmark.y * h)
            x = int(landmark.x * w)
            cv2.circle(img, (x, y), 3, (0, 255, 255), cv2.FILLED)
            
            iris_center = landmarks_points[473]
            
            min_x, max_x = 0.35, 0.55
            min_y, max_y = 0.35, 0.55
            
            normalized_x = (iris_center.x - min_x) / (max_x - min_x)
            normalized_y = (iris_center.y - min_y) / (max_y - min_y)

            screen_x = int(normalized_x * screen_w)
            screen_y = int(normalized_y * screen_h)

            smooth_x = smooth_x * (1 - damping) + screen_x * damping
            smooth_y = smooth_y * (1 - damping) + screen_y * damping

            pyautogui.moveTo(smooth_x, smooth_y)

            if left_eye[0].y - left_eye[1].y < 0.01:
                pyautogui.click()
                pyautogui.sleep(2)
                print("Mouse Clicked")

        cv2.imshow("Eye Control Mouse", img)
    key = cv2.waitKey(100)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()