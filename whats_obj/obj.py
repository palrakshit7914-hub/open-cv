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
            x1,y1,x2,y2 = r.boxes.xyxy[0]#making bounding box coordinates into integers
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            # conf = float(r.boxes.conf[0])
            conf = round(float(r.boxes.conf[0]), 2)#confidence score and round to 2 decimal places

            class_id = int(r.boxes.cls[0])#class id of the object detected 
            class_name = model.names[class_id]#class name of the object detected



    cv2.imshow("Object Detection",img)
    if cv2.waitKey(10) == 27:
        break
webcam.release()
cv2.destroyAllWindows()