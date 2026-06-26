import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

webcam = cv2.VideoCapture(0)
while True:
    _,img = webcam.read()
    img = cv2.flip(img,1)#mirror the image

    results = model(img,stream=True)#optimized img for webcam
    for r in results:
        img = r.plot()
        for box in r.boxes.xyxy:
            x1,y1,x2,y2 = r.boxes.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # conf = r.boxes.conf[0]

    cv2.imshow("Object Detection",img)
    if cv2.waitKey(10) == 27:
        break
webcam.release()
cv2.destroyAllWindows()