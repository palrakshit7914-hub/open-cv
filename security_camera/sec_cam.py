import cv2
import winsound 

webcam = cv2.VideoCapture(0)
while True:
    _,img1 = webcam.read()
    _,img2 = webcam.read()
    diff = cv2.absdiff(img1,img2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Security Camera",gray)
    if cv2.waitKey(10) == 27:
        break
webcam.release()
cv2.destroyAllWindows()