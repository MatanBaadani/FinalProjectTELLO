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

# Set device to GPU if available, otherwise CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Depth Anything model
encoder = 'vitl'
depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE)
depth_anything.eval()

# Ensure dimensions are multiples of 14 (e.g., 224, 280)
input_height = 224
input_width = 224

# Transformation pipeline for depth estimation
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

# Function to detect open doors in the depth map
def detect_open_door(depth_resized):
    _, thresh = cv2.threshold(depth_resized, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 7000 < area < 40000:  # Adjust thresholds based on expected door size
            x, y, w, h = cv2.boundingRect(contour)
            if h > w:
                cv2.rectangle(depth_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
                return True
    return False

# Main loop for drone control, depth map generation, and face recognition
while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get the frame from the drone's camera
    img = me.get_frame_read().frame
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

    # Depth map generation
    img_resized = cv2.resize(img, (input_width, input_height))
    img_rgb_resized = img_resized / 255.0
    img_transformed = transform({'image': img_rgb_resized})['image']
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(img_tensor)

    depth_resized = F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[0, 0]
    depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
    depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

    # Door detection (optional for debug)
    detect_open_door(depth_resized)

    # Display the outputs
    img_display = cv2.resize(img_rgb, (360, 360))
    cv2.imshow("Drone Camera - Face Detection", img_display)
    cv2.imshow("Depth Map", depth_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
me.streamoff()
cv2.destroyAllWindows()
