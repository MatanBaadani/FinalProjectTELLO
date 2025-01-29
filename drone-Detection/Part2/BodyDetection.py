import cv2
from cvzone.PoseModule import PoseDetector

detector = PoseDetector()
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

