import cv2
face_cascade = cv2.cascasdeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
while True:
    _,img=webcam.read()
    
