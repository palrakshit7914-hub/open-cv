import cv2
import mediaPipe as mp
import pyautogui

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    cv2.imshow("Eye Control Mouse", img)
    key = cv2.waitKey(100)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()