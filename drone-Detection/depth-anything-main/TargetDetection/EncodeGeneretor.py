import cv2
import face_recognition
import os
import pickle

folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
targets = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    targets.append(os.path.splitext(path)[0])
    print(os.path.splitext(path)[0])


def findEncoding(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncoding(imgList)
encodeListKnownWithTargets = [encodeListKnown, targets]

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithTargets, file)
file.close()
print("file saved")
