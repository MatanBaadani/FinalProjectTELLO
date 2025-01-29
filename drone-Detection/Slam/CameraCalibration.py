import numpy as np
import cv2
import glob

chessboardSize = (12, 8)
frameSize = (1440, 1080)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

objPoints = []
imgPoints = []

images = glob.glob('*.png')
for image in images:
    print(image)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)
    if ret:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners)

        cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("Camera Calibration: ", ret)
print("Camera Matrix: ", cameraMatrix)
print("Distortion Vectors: ", dist)
print("Rotation Vectors: ", rvecs)
print("Translation Vectors: ", tvecs)

img = cv2.imread('opencv_frame_1.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('caliResult.png', dst)

mean_error = 0
for i in range(len(objPoints)):
    imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2) / len(imgPoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objPoints)))
