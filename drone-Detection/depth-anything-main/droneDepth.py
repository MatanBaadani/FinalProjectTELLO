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

# Initialize drone and connect
kp.init()
me = Tello()
me.connect()
print(me.get_battery())
me.streamon()
counter = 183

# Set device to CPU as GPU is not available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the Depth Anything model
encoder = 'vitl'
depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE)
depth_anything.eval()

# Ensure dimensions are multiples of 14 (e.g., 224, 280)
input_height = 448
input_width = 518

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


def getKeyboardInput(depth_image):
    global counter
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

    # Save the depth image when 'z' is pressed
    if kp.getKey('z'):
        cv2.imwrite(f'../Resources/Images/frame_{counter}.jpg', depth_image)
        counter += 1
        time.sleep(0.4)

    return [lr, fb, ud, yv]

depth_resized = np.zeros((input_height, input_width), dtype=np.uint8)  # All zeros initially

while True:
    # Get the keyboard inputs and depth image
    vals = getKeyboardInput(depth_resized)
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

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
    depth_resized = F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[0, 0]
    depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
    depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

    # Display the depth map and drone camera feed
    print(np.sum(depth_resized))
    cv2.imshow("Depth Map", depth_resized)
    img = cv2.resize(img, (240, 240))
    cv2.imshow("Drone Camera", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

me.streamoff()
cv2.destroyAllWindows()

