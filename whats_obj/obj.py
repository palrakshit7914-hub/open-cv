import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

webcam = cv2.VideoCapture(0)

while True:
    _,img = webcam.read()
    img = cv2.flip(img,1)
    h, w, _ = img.shape
    screen_cx, screen_cy = w // 2, h // 2

    results = model(img,conf=0.40, iou=0.60,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes.xyxy:
            x1,y1,x2,y2 = boxes.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            # conf = float(boxes.conf[0])
            conf = round(float(boxes.conf[0]), 2)
            class_id = int(r.boxes.cls[0])  
            class_name = model.names[class_id]

            # if class_name == "person":
            #     continue 

            box_area = (x2 - x1) * (y2 - y1)
            total_screen_area = w * h
            if box_area / total_screen_area < 0.02: 
                continue

            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2
            distance_from_center = ((obj_cx - screen_cx) ** 2 + (obj_cy - screen_cy) ** 2) ** 0.5

            if distance_from_center < (w * 0.35):
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                label = f"{class_name} {conf}"
                cv2.putText(img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Object Detection",img)
    if cv2.waitKey(10) == 27:
        break
webcam.release()
cv2.destroyAllWindows()


