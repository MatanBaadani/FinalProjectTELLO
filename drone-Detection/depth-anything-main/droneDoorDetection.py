from djitellopy import Tello
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import time
import cvzone
import pickle
import face_recognition

# Load face recognition data
file = open('TargetDetection/EncodeFile.p', 'rb')
encodeListKnownWithTargets = pickle.load(file)
file.close()
encodeListKnown, targets = encodeListKnownWithTargets

# Initialize drone and connect
me = Tello()
me.connect()
print("Battery:", me.get_battery())
me.streamon()

# Set device to CPU as GPU is not available
DEVICE = 'cpu'

# Load the Depth Anything model

# Ensure dimensions are multiples of 14 (e.g., 224, 280)
input_height = 224
input_width = 224

# Initialize PID controller for yaw axis
yaw_PID = cvzone.PID([0.22, 0, 0.1], 480 // 2)


# target detection fucntion

def rotate_and_recognize_faces(rotation_speed, rotation_time, encodeListKnown, targets):
    start_time = time.time()

    while time.time() - start_time < rotation_time:
        # Rotate the drone
        me.send_rc_control(0, 0, 0, rotation_speed)

        # Get the frame from the drone's camera
        img = me.get_frame_read().frame
        img = cv2.resize(img, (480, 480))
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
                top, right, bottom, left = faceLoc
                cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                time.sleep(1)
                me.send_rc_control(0, 0, 0, 0)
                return
        # Display the depth map and face recognition
        cv2.imshow("Drone Camera", img_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    me.send_rc_control(0, 0, 0, 0)  # Stop rotation


# function to center the target
def center_target_with_pid(me, encodeListKnown, targets, yaw_PID, pid_duration=20):
    """
    Uses PID control to center the drone on the detected target face.

    Parameters:
    - me: The drone object.
    - encodeListKnown: The list of known face encodings.
    - targets: Corresponding names of the known faces.
    - yaw_PID: The PID controller object for yaw adjustments.
    - pid_duration: The maximum duration (in seconds) to run the PID loop.

    Returns:
    - None
    """
    pid_start_time = time.time()

    while time.time() - pid_start_time < pid_duration:
        # Get the frame from the drone's camera
        img = me.get_frame_read().frame
        img = cv2.resize(img, (480, 480))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Face recognition
        faceCurFrame = face_recognition.face_locations(img_rgb)
        encodeCurFrame = face_recognition.face_encodings(img_rgb, faceCurFrame)

        center_x = None  # Initialize center_x to track the target position

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            best_match_index = np.argmin(faceDis)
            if matches[best_match_index]:
                name = targets[best_match_index]
                print(f"Detected: {name}")
                top, right, bottom, left = faceLoc
                center_x = (left + right) // 2  # Get the center x position of the detected face
                cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # If a target is detected, adjust the yaw
        if center_x is not None:
            xVal = int(yaw_PID.update(center_x))
            me.send_rc_control(0, 0, 0, xVal)  # Adjust yaw to center the target
        else:
            me.send_rc_control(0, 0, 0, 0)  # Stop if no target is detected

        # Show images
        cv2.imshow("Drone Camera", img_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop drone movement
    me.send_rc_control(0, 0, 0, 0)


# Take off
me.takeoff()
# Go up 50 cm
me.move_up(50)
rotation_speed = 25
# Start a 360-degree rotation

# Rotate to the first detected door


rotate_and_recognize_faces(rotation_speed=-25, rotation_time=30, encodeListKnown=encodeListKnown, targets=targets)
center_target_with_pid(me, encodeListKnown, targets, yaw_PID, pid_duration=20)

# Stop and land the drone
me.send_rc_control(0, 0, 0, 0)
me.land()
me.streamoff()
cv2.destroyAllWindows()
