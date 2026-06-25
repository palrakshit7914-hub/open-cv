import cv2
import os
from ultralytics import YOLO


webcam = cv2.VideoCapture(0)
while True:
    _,img = webcam.read()
    cv2.imshow("Object Detection",img)
    if cv2.waitKey(10) == 27:
        break
webcam.release()
cv2.destroyAllWindows()