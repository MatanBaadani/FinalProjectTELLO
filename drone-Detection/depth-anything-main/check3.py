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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Depth Anything model
encoder = 'vits'
depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE)
depth_anything.eval()

# Ensure dimensions are multiples of 14 (e.g., 224, 280)
input_height = 224
input_width = 224

# Transformation pipeline
transform = Compose([
    Resize(
        width=input_width,
        height=input_height,
        resize_target=False,
        keep_aspect_ratio=False,  # Disable aspect ratio to enforce fixed size
        ensure_multiple_of=14,  # Ensure dimensions are multiples of 14
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_AREA,  # Faster interpolation method
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])


# Initialize PID controller for yaw axis


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


# Function to detect doors during rotation
def detect_open_door(depth_resized, uppersize, lower=30):
    # Threshold the depth map to isolate potential door areas
    _, thresh = cv2.threshold(depth_resized, lower, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_door = None  # Store the position of the detected door
    for contour in contours:
        area = cv2.contourArea(contour)
        if 7000 < area < uppersize:  # Adjust thresholds based on expected door size
            x, y, w, h = cv2.boundingRect(contour)
            if h > 1.5 * w:
                detected_door = (x + w // 2, y + h // 2)  # Center of the door
                cv2.rectangle(depth_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return detected_door


# scaning function
def detect_doors_during_rotation(me, depth_anything, transform, DEVICE, input_width, input_height, rotation_speed=25,
                                 rotation_time=30, consistency_threshold=2.0, detection_cooldown=3.0,
                                 upper_bound=30000):
    """
    Rotates the drone and detects doors based on depth map.

    Parameters:
    - me: The drone object.
    - depth_anything: The depth prediction model.
    - transform: Image transformation function.
    - DEVICE: The device for PyTorch (e.g., 'cuda' or 'cpu').
    - input_width: The width to resize the input image for depth prediction.
    - input_height: The height to resize the input image for depth prediction.
    - rotation_speed: The speed at which the drone rotates.
    - rotation_time: The total time for the drone to complete a rotation.
    - consistency_threshold: Time in seconds for consistent detection to count as a door.
    - detection_cooldown: Cooldown period to avoid multiple detections of the same door.
    - upper_bound: The threshold parameter for the detect_open_door function.

    Returns:
    - detection_times: List of times (in seconds from the start) when doors were detected.
    """
    start_time = time.time()

    # List to store detection times
    detection_times = []
    detection_start_time = None
    last_detection_time = None

    while time.time() - start_time < rotation_time:
        # Rotate the drone
        me.send_rc_control(0, 0, 0, rotation_speed)

        # Get the frame from the drone's camera
        img = me.get_frame_read().frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and process the image for depth estimation
        img_resized = cv2.resize(img, (input_width, input_height))
        img_rgb = img_resized / 255.0

        # Transform and prepare for the model
        img_transformed = transform({'image': img_rgb})['image']
        img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)

        # Predict depth
        with torch.no_grad():
            depth = depth_anything(img_tensor)

        # Resize depth map to original dimensions
        depth_resized = F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[
            0, 0]
        depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
        depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

        # Check for door detection
        detected_door = detect_open_door(depth_resized, upper_bound)
        if detected_door:
            if detection_start_time is None:
                detection_start_time = time.time()
            else:
                elapsed_time = time.time() - detection_start_time
                if elapsed_time >= consistency_threshold:
                    current_time = time.time() - start_time
                    if last_detection_time is None or (current_time - last_detection_time) >= detection_cooldown:
                        detection_times.append(round(current_time, 2))
                        last_detection_time = current_time
                    detection_start_time = None
        else:
            detection_start_time = None

        # Display the depth map
        cv2.imshow("Depth Map", depth_resized)
        img_original = cv2.resize(img, (360, 360))
        cv2.imshow("Drone Camera", img_original)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return detection_times


def center_and_move_forward(me, depth_anything, transform, DEVICE, input_width, input_height, yaw_PID, pid_duration=20,
                            forward_speed=20, loss_duration=1, upper_bound=40000, time_forwrd=3, lower=30):
    """
    Uses PID control to center the drone on the detected door and then moves forward,
    continuing to use PID for corrections until the door is no longer detected.

    Parameters:
    - me: The drone object.
    - depth_anything: The depth prediction model.
    - transform: Image transformation function.
    - DEVICE: The device for PyTorch (e.g., 'cuda' or 'cpu').
    - input_width: The width to resize the input image for depth prediction.
    - input_height: The height to resize the input image for depth prediction.
    - yaw_PID: The PID controller object for yaw adjustments.
    - pid_duration: The maximum duration (in seconds) to run the initial PID centering loop.
    - forward_speed: The speed at which the drone moves forward after centering.
    - loss_duration: The duration (in seconds) of continuous door loss required to stop moving forward.
    - upper_bound: The threshold parameter for the detect_open_door function.

    Returns:
    - None
    """
    pid_start_time = time.time()
    last_detection_time = time.time()

    # Initial PID centering
    while time.time() - pid_start_time < pid_duration:
        img = me.get_frame_read().frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and process the image for depth estimation
        img_resized = cv2.resize(img, (input_width, input_height))
        img_rgb = img_resized / 255.0
        img_transformed = transform({'image': img_rgb})['image']
        img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)

        # Predict depth
        with torch.no_grad():
            depth = depth_anything(img_tensor)
        depth_resized = F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[
            0, 0]
        depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
        depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

        # Check for door detection
        detected_door = detect_open_door(depth_resized, upper_bound, lower)
        if detected_door:
            x, y = detected_door
            xVal = int(yaw_PID.update(x-25))
            me.send_rc_control(0, 0, 0, xVal)  # Adjust yaw to center the door
            last_detection_time = time.time()  # Reset last detection time
        else:
            me.send_rc_control(0, 0, 0, 0)  # Stop if no door is detected

        # Show images
        cv2.imshow("Depth Map", depth_resized)
        img_original = cv2.resize(img, (360, 360))
        cv2.imshow("Drone Camera", img_original)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Move forward with continued PID adjustments
    while True:
        img = me.get_frame_read().frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and process the image for depth estimation
        img_resized = cv2.resize(img, (input_width, input_height))
        img_rgb = img_resized / 255.0
        img_transformed = transform({'image': img_rgb})['image']
        img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)

        # Predict depth
        with torch.no_grad():
            depth = depth_anything(img_tensor)
        depth_resized = F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[
            0, 0]
        depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
        depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

        # Check for door detection
        detected_door = detect_open_door(depth_resized, upper_bound)
        if detected_door:
            x, y = detected_door
            xVal = int(yaw_PID.update(x))
            me.send_rc_control(0, forward_speed, 0, xVal)  # Move forward while adjusting yaw
            last_detection_time = time.time()  # Reset last detection time
        else:
            if time.time() - last_detection_time > loss_duration:
                break  # Stop if no door is detected for the specified loss duration
            me.send_rc_control(0, forward_speed, 0, 0)  # Move forward without yaw adjustment

        # Show images
        cv2.imshow("Depth Map", depth_resized)
        img_original = cv2.resize(img, (360, 360))
        cv2.imshow("Drone Camera", img_original)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop drone movement

    me.send_rc_control(0, forward_speed, 0, 0)
    time.sleep(time_forwrd)


def rotate_to_first_detected_door(me, rotation_speed, first_door_time):
    """
    Rotates the drone to face the first detected door.
    Parameters:
    - me: The drone object.
    - rotation_speed: The speed at which the drone should rotate.
    - first_door_time: The time (in seconds) for which the drone should rotate to face the door.
    Returns:
    - None
    """
    print(f"Rotating to the first detected door at {first_door_time} seconds...")

    # Perform an initial rotation to face the door area
    me.send_rc_control(0, 0, 0, rotation_speed)
    time.sleep(first_door_time)  # Rotate for the duration of the first detection time
    me.send_rc_control(0, 0, 0, 0)  # Stop rotation


yaw_PID = cvzone.PID([0.22, 0, 0.1], input_width // 2)
# Take off
me.takeoff()
# Go up 50 cm
me.move_up(80)
rotation_speed = 25
# Start a 360-degree rotation

# Start a 360-degree rotation
detection_times = detect_doors_during_rotation(
    me=me,
    depth_anything=depth_anything,
    transform=transform,
    DEVICE=DEVICE,
    input_width=input_width,
    input_height=input_height,
    rotation_speed=rotation_speed,
    rotation_time=30,
    consistency_threshold=2.0,
    detection_cooldown=3.0,
    upper_bound=30000
)

# Rotate to the first detected door
if detection_times:
    first_door_time = detection_times[0] - 3

    rotate_to_first_detected_door(me=me, rotation_speed=rotation_speed, first_door_time=first_door_time)

    center_and_move_forward(me, depth_anything, transform, DEVICE, input_width, input_height, yaw_PID,
                            pid_duration=12, forward_speed=20, loss_duration=1, upper_bound=45000, lower=20)

detection_times = detect_doors_during_rotation(
    me=me,
    depth_anything=depth_anything,
    transform=transform,
    DEVICE=DEVICE,
    input_width=input_width,
    input_height=input_height,
    rotation_speed=rotation_speed,
    rotation_time=30,
    consistency_threshold=2.0,
    detection_cooldown=3.0,
    upper_bound=30000
)
if detection_times:
    first_door_time = detection_times[0] - 3

    rotate_to_first_detected_door(me=me, rotation_speed=rotation_speed, first_door_time=first_door_time)

    center_and_move_forward(me, depth_anything, transform, DEVICE, input_width, input_height, yaw_PID,
                            pid_duration=12, forward_speed=20, loss_duration=2, upper_bound=40000, time_forwrd=3)
yaw_PID = cvzone.PID([0.22, 0, 0.1], 480 // 2)
rotate_and_recognize_faces(rotation_speed=-25, rotation_time=30, encodeListKnown=encodeListKnown, targets=targets)
center_target_with_pid(me, encodeListKnown, targets, yaw_PID, pid_duration=12)
# Stop and land the drone
me.send_rc_control(0, 0, 0, 0)
me.land()
me.streamoff()
cv2.destroyAllWindows()
