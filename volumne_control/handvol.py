import cv2
import mediapipe as mp
import pyautogui

webcam = cv2.VideoCapture(0)
while True:
    _,img = webcam.read()
    cv2.imshow("Hand Volume Control using Python", img)
    waitKey = cv2.waitKey(10)
    if waitKey == 27:
        break