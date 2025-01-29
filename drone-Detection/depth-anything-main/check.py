from djitellopy import Tello
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import KeyPressMoudle as kp
import time
import pickle
import face_recognition

# Initialize drone and connect
kp.init()
me = Tello()
me.connect()
print("Battery:", me.get_battery())
me.streamon()

# Set device to CPU as GPU is not available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Ensure dimensions are multiples of 14 (e.g., 224, 280)
input_height = 224
input_width = 224

# Load face recognition data
file = open('TargetDetection/EncodeFile.p', 'rb')
encodeListKnownWithTargets = pickle.load(file)
file.close()
encodeListKnown, targets = encodeListKnownWithTargets
print("Known Targets:", targets)

# Function to get keyboard input for drone control
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if kp.getKey("LEFT"):
        lr = -speed
    elif kp.getKey("RIGHT"):
        lr = speed

    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed

    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed

    if kp.getKey("a"):
        yv = -speed
    elif kp.getKey("d"):
        yv = speed

    if kp.getKey("q"):
        me.land()
        time.sleep(3)
    if kp.getKey("e"):
        me.takeoff()

    return [lr, fb, ud, yv]

# Function to detect open doors in the depth mapq

# Variables for door detection timin

# Main loop for drone control, door detection, and face recognition
while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get the frame from the drone's camera
    img = me.get_frame_read().frame
    img=cv2.resize(img,(360,360))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face recognition
    faceCurFrame = face_recognition.face_locations(img_rgb)
    encodeCurFrame = face_recognition.face_encodings(img_rgb, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        best_match_index = np.argmin(faceDis)
        if matches[best_match_index]:
            name = targets[best_match_index]
            print(f"Detected: {name}")
            print(faceLoc)
            top, right, bottom, left = faceLoc
            cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img_rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the depth map and face recognition
    cv2.imshow("Drone Camera", img_rgb)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
me.streamoff()
cv2.destroyAllWindows()
