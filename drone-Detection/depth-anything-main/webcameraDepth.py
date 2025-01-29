import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import time
from ultralytics import YOLO
import math
import cvzone

# Choose encoder
encoder = 'vitl'
video_path = 0

# Set device to CPU as GPU is not available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
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

# Video capture and writer setup
cap = cv2.VideoCapture(video_path)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 480))

# Variables to track detection consistency
last_door_contour = None
detection_start_time = None
detection_variable = 0  # Will be set to 1 if a door is detected consistently for 2 seconds
detection_times = []  # List to store detection timestamps

# Timer starting from 0
start_time = time.time()

model= YOLO("best.pt").to(DEVICE)  # Load the YOLO model from the specified path
class_names = ["obstacle"]
confidence = 0.65  # Confidence threshold for object detection

# Get the name of each GPU
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

        # Perform YOLO object detection


def get_object_list_yolo(model, img, class_names, confidence, draw=True,filter= False, filter_obj=[]):
    """
    Performs YOLO object detection on the given frame.
    """
    results = model(img, stream=False, verbose=False)  # Run the YOLO model on the frame
    object_list = []  # List to store detected objects
    for result in results:  # Loop through each detection result
        boxes = result.boxes  # Get bounding boxes from the detection result
        for box in boxes:  # Loop through each bounding box
            conf = math.ceil((box.conf[0] * 100)) / 100  # Get the confidence score and round it to 2 decimal places
            if conf > confidence:  # Check if the confidence score is above the threshold
                class_name = class_names[int(box.cls[0])]  # Get the class name from the detection result
                if filter and class_name not in filter_obj:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
                w, h = x2 - x1, y2 - y1  # Calculate width and height of the bounding box
                center = x1 + (w // 2), y1 + (h // 2)  # Calculate the center of the bounding box

                # Add the detected object to the list
                object_list.append({"bbox": (x1, y1, w, h),
                                    "center": center,
                                    "conf": conf,
                                    "class": class_name})

                if draw:  # Check if drawing is enabled
                    cvzone.cornerRect(img, (x1, y1, w, h))  # Draw a rectangle around the detected object
                    cvzone.putTextRect(img, f'{class_name} {conf}',
                                           (max(0, x1), max(35, y1)), scale=1, thickness=1)  # Draw the class name and confidence score


while cap.isOpened():
    ret, raw_image = cap.read()

    if not ret:
        break

    # Resize raw image early to save computation
    raw_image = cv2.resize(raw_image, (input_width, input_height))

    # Convert image to RGB and normalize
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]

    # Apply transformations and prepare for model input
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    # Predict depth with no gradients for speed
    with torch.no_grad():
        depth = depth_anything(image)

    # Resize depth map to original dimensions
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)

    # Threshold the depth map to isolate the open door (darker regions)
    _, thresholded = cv2.threshold(depth, 50, 255, cv2.THRESH_BINARY_INV)

    #depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # Resize back to the original frame size for saving
    #depth_color = cv2.resize(depth, (640, 480))

    # Resize the depth map to match output video resolution
    depth_resized = cv2.resize(depth, (640, 480))

    # Convert single-channel depth map to 3-channel grayscale
    depth_3channel = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
    get_object_list_yolo(model, depth_3channel, class_names, confidence, draw=True, filter=False,
                         filter_obj=[])  # Detect objects and draw on the frame
    # Save the depth map as a video
    #out_video.write(depth_3channel)

    # Display the depth map
    cv2.imshow('Depth Anything', depth_3channel)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
#out_video.release()
cv2.destroyAllWindows()