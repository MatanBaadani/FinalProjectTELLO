import pickle
import cv2
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 560)
cap.set(4, 480)

file = open('EncodeFile.p', 'rb')
encodeListKnownWithTargets = pickle.load(file)
file.close()
encodeListKnown, targets = encodeListKnownWithTargets
print(targets)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("match", matches)
        print("faceDis", faceDis)

    cv2.imshow("Face Attendance", img)
    cv2.waitKey(1)
