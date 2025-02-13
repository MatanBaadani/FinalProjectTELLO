import cv2
import time
import torch
import numpy as np
import torch.nn.functional as F
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Constants
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
VIDEO_PATH = "1.mp4"  # Change this to your video path


# Initialize Depth Anything model
def initialize_depth_anything():
    transform = Compose([
        Resize(width=INPUT_WIDTH, height=INPUT_HEIGHT, resize_target=False, keep_aspect_ratio=False,
               ensure_multiple_of=14, image_interpolation_method=cv2.INTER_AREA),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(device)
    return model.eval(), transform, device


# Define f1
def f1(img, model, transform, DEVICE):
    imgcolor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_size = (INPUT_WIDTH, INPUT_HEIGHT)
    img_resized = cv2.resize(imgcolor, input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img_transformed = transform({'image': img_resized})['image']
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.no_grad():
        depth = model(img_tensor)

    depth_resized = F.interpolate(depth.unsqueeze(0), size=input_size, mode='bilinear', align_corners=False)[0, 0]
    depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
    depth_final = torch.clamp(depth_resized * 255, 0, 255).to(torch.uint8).cpu().numpy()
    return depth_final


# Define f2
def f2(img, model, transform, DEVICE):
    imgcolor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(imgcolor, (INPUT_WIDTH, INPUT_HEIGHT))
    img_rgb = img_resized / 255.0
    img_transformed = transform({'image': img_rgb})['image']
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = model(img_tensor)

    depth_resized = F.interpolate(depth[None], (INPUT_HEIGHT, INPUT_WIDTH), mode='bilinear', align_corners=False)[0, 0]
    depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
    depth_resized = depth_resized.cpu().numpy().astype(np.uint8)
    return depth_resized


# Function to measure average processing time per frame
def measure_avg_time(video_path, func, model, transform, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    total_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        start_time = time.time()
        _ = func(frame, model, transform, device)  # Run function on frame
        total_time += (time.time() - start_time)
        frame_count += 1

    cap.release()

    avg_time_per_frame = (total_time / frame_count) * 1000 if frame_count > 0 else 0  # Convert to ms
    return avg_time_per_frame


# Run the test
model, transform, device = initialize_depth_anything()

# Measure f1
avg_time_f1 = measure_avg_time(VIDEO_PATH, f1, model, transform, device)
print(f"Average processing time per frame for f1: {avg_time_f1:.2f} ms")

# Measure f2
avg_time_f2 = measure_avg_time(VIDEO_PATH, f2, model, transform, device)
print(f"Average processing time per frame for f2: {avg_time_f2:.2f} ms")
