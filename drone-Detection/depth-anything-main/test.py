import time
import torch
import cv2
from ultralytics import YOLO
import numpy as np

# Create a function to time the YOLO inference on a given device
def run_inference_on_device(device, model, image):
    model.to(device)
    image = image.to(device)  # Move image to the selected device

    # Start time for inference
    start_time = time.time()

    # Run inference (disable gradient calculations for inference)
    with torch.no_grad():
        results = model(image, stream=False, verbose=False)

    # End time for inference
    end_time = time.time()

    # Calculate and return the inference time
    inference_time = end_time - start_time
    return inference_time, results


# Function to process the video and compare CPU vs GPU performance
def compare_performance_on_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the video frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Load YOLO model
    model = YOLO("best.pt").to("cpu")  # Load model on CPU initially

    # Variables to store total time
    cpu_total_time = 0
    gpu_total_time = 0
    frame_count = 0

    # Process video frames on CPU
    print("Processing video on CPU...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize image to 640x640 (ensure it is divisible by 32)
        image_resized = cv2.resize(frame, (640, 640))

        # Prepare image (convert to RGB, normalize, etc.)
        image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]

        # Convert the image to a tensor and change shape from HWC (Height, Width, Channels) to BCHW (Batch, Channels, Height, Width)
        image = torch.tensor(image).unsqueeze(0).float()  # Convert to tensor and add batch dimension
        image = image.permute(0, 3, 1, 2)  # Change shape to (1, 3, 640, 640)

        # Measure time on CPU
        cpu_time, _ = run_inference_on_device("cpu", model, image)
        cpu_total_time += cpu_time

        # Measure time on GPU (if available)
        if torch.cuda.is_available():
            gpu_time, _ = run_inference_on_device("cuda", model, image)
            gpu_total_time += gpu_time

    cap.release()

    # Calculate the total time for CPU and GPU processing
    print(f"Processed {frame_count} frames")
    print(f"Total time on CPU: {cpu_total_time:.4f} seconds")
    if torch.cuda.is_available():
        print(f"Total time on GPU: {gpu_total_time:.4f} seconds")
    else:
        print("No GPU available for comparison.")

    # Calculate and print average time per frame
    print(f"Average time per frame on CPU: {cpu_total_time / frame_count:.4f} seconds")
    if torch.cuda.is_available():
        print(f"Average time per frame on GPU: {gpu_total_time / frame_count:.4f} seconds")


# Example: Compare performance on a video
video_path = "1.mp4"  # Replace with your video file path
compare_performance_on_video(video_path)
