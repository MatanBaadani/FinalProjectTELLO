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
import threading
import math
from ultralytics import YOLO

# Global variables for frame processing
img = None  # Global variable to store the latest frame captured from the drone
drone = None  # Global variable for drone control

frame_lock = threading.Lock()  # Lock to control access to the shared 'img' variable
frame_ready = threading.Event()  # Event to signal when a new frame is ready to be processed
exit_flag = threading.Event()  # Event to signal when to exit the program

# Ensure dimensions are multiples of 14 (e.g., 224, 280)
input_height = 224
input_width = 224


def initialize_drone():
    """
    Initializes the drone by connecting and starting the video stream.
    """
    me = Tello()  # Create a Tello drone object
    me.connect()  # Connect to the drone
    print(me.get_battery())  # Print the current battery level
    me.streamon()  # Turn on the video stream from the drone
    return me  # Return the drone object


def initialize_depth_anything(model_path="../Yolo_Weights/yolov10n.pt"):
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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the Depth Anything model
    encoder = 'vits'
    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE)
    return depth_anything.eval(), transform, DEVICE


def capture_frames(drone):
    """
    Continuously captures frames from the drone and updates the global 'img' variable.
    This function runs in a separate thread.
    """
    global img  # Declare 'img' as global to modify it within this function
    while not exit_flag.is_set():  # Check for exit flag to stop the loop
        frame = drone.get_frame_read().frame  # Capture a frame from the drone's camera
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert the frame from RGB to BGR (OpenCV format)

        # cv2.imshow("Drone Feed", frame)
        with frame_lock:  # Acquire the lock to safely update the shared 'img' variable
            img = frame.copy()  # Update the global 'img' with the new frame
            frame_ready.set()  # Signal that a new frame is ready for processing
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
            exit_flag.set()
            break


def process_frames(drone, model, transform, DEVICE, yaw_PID, model_obstacle, class_names, confidence, rotation_time=30,
                   rotation_speed=25,consistency_threshold=1.8, detection_cooldown=3.0, upper_bound=30000, pid_duration=20,
                   forward_speed=20, loss_duration=1, time_forwrd=1.5, lower=30):
    global img  # Declare 'img' as global to access it within this function
    while not exit_flag.is_set():  # Check for exit flag to stop the loop
        start_time = time.time()

        # List to store detection times
        detection_times = []
        detection_start_time = None
        last_detection_time = None
        # Get the frame from the drone's camera
        while time.time() - start_time < rotation_time:
            frame_ready.wait()  # Wait until a new frame is ready for processing
            frame_ready.clear()  # Clear the event to reset it for the next frame

            with frame_lock:  # Acquire the lock to safely read the shared 'img' variable
                if img is not None:  # Check if 'img' is not None to avoid errors
                    depth_img = img.copy()  # Copy the frame for processing
            drone.send_rc_control(0, 0, 0, rotation_speed)
            RGB_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
            # Resize and process the image for depth estimation
            img_resized = cv2.resize(RGB_img, (input_width, input_height))
            img_rgb = img_resized / 255.0
            img_transformed = transform({'image': img_rgb})['image']
            img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                depth = model(img_tensor)

            # Resize depth map to original dimensions
            depth_resized = \
                F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[
                    0, 0]
            depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
            depth_resized = depth_resized.cpu().numpy().astype(np.uint8)
            depth_3channel = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
            detected_doors = get_object_list_yolo(model_obstacle, depth_3channel, class_names, confidence, draw=True,
            filter=False,filter_obj=[])
            # Check for door detection
            if detected_doors:
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
            cv2.imshow("Depth Map", depth_3channel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break
        if detection_times:
            first_door_time = detection_times[0] - 3
            print(f"Rotating to the first detected door at {first_door_time} seconds...")
            # Perform an initial rotation to face the door area
            drone.send_rc_control(0, 0, 0, rotation_speed)
            time.sleep(first_door_time)  # Rotate for the duration of the first detection time
            drone.send_rc_control(0, 0, 0, 0)  # Stop rotation
            print(detection_times)
        pid_start_time = time.time()
        last_detection_time = time.time()
        while time.time() - pid_start_time < pid_duration:
            img = drone.get_frame_read().frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize and process the image for depth estimation
            img_resized = cv2.resize(img, (input_width, input_height))
            img_rgb = img_resized / 255.0
            img_transformed = transform({'image': img_rgb})['image']
            img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                depth = model(img_tensor)
            depth_resized = \
                F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[
                    0, 0]
            depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
            depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

            # Check for door detection

            depth_3channel = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
            detected_doors = get_object_list_yolo(model_obstacle, depth_3channel, class_names, confidence, draw=True,
            filter=False,filter_obj=[])
            if detected_doors:
                x, y = detected_doors[0]["center"]
                xVal = int(yaw_PID.update(x))
                drone.send_rc_control(0, 0, 0, xVal)  # Adjust yaw to center the door
                last_detection_time = time.time()  # Reset last detection time
            else:
                drone.send_rc_control(0, 0, 0, 0)  # Stop if no door is detected

            # Show images
            cv2.imshow("Depth Map", depth_3channel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break

        # Move forward with continued PID adjustments
        while True:
            img = drone.get_frame_read().frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize and process the image for depth estimation
            img_resized = cv2.resize(img, (input_width, input_height))
            img_rgb = img_resized / 255.0
            img_transformed = transform({'image': img_rgb})['image']
            img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE)

            # Predict depth
            with torch.no_grad():
                depth = model(img_tensor)
            depth_resized = \
                F.interpolate(depth[None], (input_height, input_width), mode='bilinear', align_corners=False)[
                    0, 0]
            depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min()) * 255.0
            depth_resized = depth_resized.cpu().numpy().astype(np.uint8)

            # Check for door detection
            depth_3channel = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
            detected_doors = get_object_list_yolo(model_obstacle, depth_3channel, class_names, confidence, draw=True,
                                                  filter=False,
                                                  filter_obj=[])
            if detected_doors:
                x, y = detected_doors[0]["center"]
                xVal = int(yaw_PID.update(x))
                drone.send_rc_control(0, forward_speed, 0, xVal)  # Move forward while adjusting yaw
                last_detection_time = time.time()  # Reset last detection time
                # area_0 = area
            else:
                if time.time() - last_detection_time > loss_duration:
                    break  # Stop if no door is detected for the specified loss duration
                drone.send_rc_control(0, forward_speed, 0, 0)  # Move forward without yaw adjustment

            # Show images
            cv2.imshow("Depth Map", depth_3channel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break

        # Stop drone movement

        drone.send_rc_control(0, forward_speed, 0, 0)
        time.sleep(time_forwrd)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
            exit_flag.set()
            break


def get_object_list_yolo(model, img, class_names, confidence, draw=True, filter=False, filter_obj=[]):
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
                detected_door = center
                # Add the detected object to the list
                object_list.append({"bbox": (x1, y1, w, h),
                                    "center": center,
                                    "conf": conf,
                                    "class": class_name})

                if draw:  # Check if drawing is enabled
                    cvzone.cornerRect(img, (x1, y1, w, h))  # Draw a rectangle around the detected object
                    cvzone.putTextRect(img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1)  # Draw the class name and confidence score
    return object_list

def main():
    """
    Main function to initialize the drone, YOLO model, and start the threads for capturing and processing frames.
    """
    drone = initialize_drone()  # Initialize the drone
    model_obstacle = YOLO("best.pt")  # Load the YOLO model from the specified path
    class_names = ["obstacle"]
    confidence = 0.80  # Confidence threshold for object detection
    model, transform, DEVICE = initialize_depth_anything()
    drone.takeoff()
    # Go up 50 cm
    drone.move_up(50)
    rotation_speed = 25
    yaw_PID = cvzone.PID([0.22, 0, 0.1], input_width // 2)

    # Start frame capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(drone,))
    # Create a new thread for the 'capture_frames' function, passing the drone object as an argument.
    # This thread will continuously capture frames from the drone and update the global 'img' variable.

    # Start processing thread for YOLO detection
    process_thread = threading.Thread(target=process_frames, args=(
        drone, model, transform, DEVICE, yaw_PID, model_obstacle, class_names, confidence))
    # Create another thread for the 'process_frames' function, passing the YOLO model, class names, and confidence as arguments.
    # This thread will process the captured frames for object detection using the YOLO model.

    capture_thread.start()  # Start the frame capture thread

    process_thread.start()  # Start the processing thread

    capture_thread.join()  # Wait for the frame capture thread to finish
    # This ensures that the main program waits for the 'capture_frames' thread to complete before proceeding.
    # Since the thread runs in an infinite loop, it will only finish if the loop is broken (e.g., by pressing 'q').

    process_thread.join()  # Wait for the processing thread to finish
    # Similarly, this waits for the 'process_frames' thread to complete before continuing.
    # This also runs in an infinite loop and will only finish if the loop is broken (e.g., by pressing 'q').q

    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()  # Call the main function to start the program
    # This ensures the program starts by initializing the drone, YOLO model, and starting the threads.
