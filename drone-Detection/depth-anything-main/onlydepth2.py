import threading
import time
import cv2
import torch
import numpy as np
import math
from djitellopy import Tello
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from ultralytics import YOLO
import cvzone
import torch.nn.functional as F

# Constants for configuration
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.80
# Global variables0
img = None
frame_lock = threading.Lock()
frame_ready = threading.Event()
exit_flag = threading.Event()

found_user = 0
found_Target = 0

route = np.zeros((1000, 1000, 3), np.uint8)
aSpeed = 12  # Angular Speed Degrees/s  25
interval = 0.07
dInterval = 0.7
start_point = [500, 800]
d = 0
a = 90
d_reset = 0
x = 500
y = 800


def drawMap():
    global route, a, x, y, d, img, d_reset, start_point
    while not exit_flag.is_set():
        frame_ready.wait()  # Wait until a new frame is ready for processing
        frame_ready.clear()  # Clear the event to reset it for the next frame
        # Stage 1: Rotate 360 degrees and detect doors
        with frame_lock:  # Acquire the lock to safely read the shared 'img' variable
            if img is not None and d_reset != 0:
                x += int(d * math.cos(math.radians(a)))
                y += int(d * math.sin(math.radians(a)))
                print(a, d, x, y, d_reset)
                d = 0
                d_reset = 0
                door_coordinates = [int((x - start_point[0]) * 0.85 + start_point[0]),
                                    int((y - start_point[1]) * 0.85 + start_point[1])]
                cv2.circle(route, start_point, 4, (0, 0, 255), cv2.FILLED)
                cv2.rectangle(route, (door_coordinates[0] - 3, door_coordinates[1] - 6),
                              (door_coordinates[0] + 3, door_coordinates[1] + 6), (0, 255, 255), 2)
                cv2.line(route, start_point, (x, y), (0, 0, 255), 2)
                if door_coordinates[0]>=start_point[0] and door_coordinates[1]>=start_point[1]:
                    cv2.rectangle(route,start_point,
                                  (door_coordinates[0] + 3, door_coordinates[1] + 6), (255, 255,0), 1)
                elif door_coordinates[0]>=start_point[0] and door_coordinates[1]<=start_point[1]:
                    cv2.rectangle(route,(start_point[0],door_coordinates[1] - 6),
                                  (door_coordinates[0] + 3,start_point[1]), (255, 255,0), 1)
                elif door_coordinates[0]<=start_point[0] and door_coordinates[1]>=start_point[1]:
                    cv2.rectangle(route,(door_coordinates[0]-3,start_point[1]),
                                  (start_point[0], door_coordinates[1] + 6), (255, 255,0), 1)
                elif door_coordinates[0]<=start_point[0] and door_coordinates[1]<=start_point[1]:
                    cv2.rectangle(route,(door_coordinates[0] - 3, door_coordinates[1] - 6),
                                  start_point, (255, 255,0), 1)
                start_point = [x, y]
            cv2.imshow("Map", route)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break


def initialize_drone():
    """
    Initializes the Tello drone and starts the video stream.
    """
    me = Tello()
    me.connect()
    print(me.get_battery())
    me.streamon()
    return me


def initialize_depth_anything():
    """
    Initializes the Depth Anything model and returns the model, transform, and device.
    """
    transform = Compose([
        Resize(width=INPUT_WIDTH, height=INPUT_HEIGHT, resize_target=False, keep_aspect_ratio=False,
               ensure_multiple_of=14, image_interpolation_method=cv2.INTER_AREA),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_vits14').to(device)
    return model.eval(), transform, device


def capture_frames(drone):
    """
    Captures frames from the drone and updates the global 'img' variable.
    """
    global img
    while not exit_flag.is_set():
        frame = drone.get_frame_read().frame
        with frame_lock:
            img = frame.copy()
            frame_ready.set()
        time.sleep(interval)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
            exit_flag.set()
            break


def detect_doors_yolo(model, img, confidence):
    """
    Detects doors using the YOLO model.
    """
    results = model(img, stream=False, verbose=False)
    object_list = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                center = x1 + w // 2, y1 + h // 2
                # Update the list to contain only the largest object
                object_list = [{"bbox": (x1, y1, w, h), "center": center, "conf": conf, "class": "open_door",}]
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, "open_door" f'{conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return object_list


def process_frames(drone, model, transform, DEVICE, yaw_PID, model_obstacle, confidence, rotation_time=30,
                   rotation_speed=25, consistency_threshold=1.5, detection_cooldown=3.0, pid_duration=12,
                   forward_speed=20, loss_duration=1, time_forwrd=3.3):
    global img, a, d, d_reset  # Declare 'img' as global to access it within this function

    # Helper functions
    def process_frame_for_depth_optimized():
        """
        Optimized function to process an image for depth estimation.
        """
        global img

        # Predefine input size to avoid recalculating it
        input_size = (INPUT_WIDTH, INPUT_HEIGHT)

        # Resize image to the desired dimensions
        img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

        # Transform the image (assume `transform` is efficient)
        img_transformed = transform({'image': img_resized})['image']

        # Convert to tensor and move to the appropriate device
        img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE, non_blocking=True)

        # Perform inference with `torch.no_grad` for efficiency
        with torch.no_grad():
            depth = model(img_tensor)  # Shape: (N, 1, H, W)
        depth = depth.unsqueeze(0)  # Add batch dimension (N=1)

        # Interpolate depth to match input dimensions if necessary
        depth_resized = F.interpolate(depth, size=input_size, mode='bilinear', align_corners=False)[0, 0]

        # Normalize depth values to [0, 1] in-place for efficiency
        depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)

        # Scale to [0, 255] and convert to uint8 for the final output
        depth_final = (depth_resized * 255).clamp(0, 255).byte().squeeze(0).cpu().numpy()

        return depth_final

    def detect_doors(depth_img):
        depth_3channel = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
        return detect_doors_yolo(model_obstacle, depth_3channel, confidence), depth_3channel

    while not exit_flag.is_set():  # Check for exit flag to stop the loop
        start_time = time.time()

        detection_times = []
        detection_start_time = None
        last_detection_time = None

        # Stage 1: Rotate 360 degrees and detect doors
        while time.time() - start_time < rotation_time:
            frame_ready.wait()  # Wait until a new frame is ready for processing
            frame_ready.clear()  # Clear the event to reset it for the next frame

            with frame_lock:  # Acquire the lock to safely read the shared 'img' variable
                if img is not None:  # Check if 'img' is not None to avoid errors
                    depth_img = process_frame_for_depth_optimized()
            drone.send_rc_control(0, 0, 0, rotation_speed)
            objects, depth_3channel = detect_doors(depth_img)

            if objects:
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
            drone.send_rc_control(0, 0, 0, rotation_speed)
            a += aSpeed * first_door_time
            time.sleep(first_door_time)
            drone.send_rc_control(0, 0, 0, 0)
            print(detection_times)

        # Stage 2: PID control to center the drone on the first detected door
        pid_start_time = time.time()
        last_detection_time = time.time()
        while time.time() - pid_start_time < pid_duration:
            depth_img = process_frame_for_depth_optimized()
            objects, depth_3channel = detect_doors(depth_img)
            if objects:
                detected_door = objects[0]
                x_center_set, y_center = detected_door["center"]
                x_val = int(yaw_PID.update(x_center_set))
                drone.send_rc_control(0, 0, 0, x_val)
                last_detection_time = time.time()
            else:
                drone.send_rc_control(0, 0, 0, 0)

            # Show images
            cv2.imshow("Depth Map", depth_3channel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break

        # Stage 3: Move forward while adjusting yaw using PID
        while True:
            depth_img = process_frame_for_depth_optimized()
            objects, depth_3channel = detect_doors(depth_img)
            if objects:
                for object in objects:


                detected_door = objects[0]
                x_center, y_center = detected_door["center"]
                if
                    x_val = int(yaw_PID.update(x_center))
                    drone.send_rc_control(0, forward_speed, 0, x_val)
                    d += dInterval
                    last_detection_time = time.time()
                    x_set = area
                elif time.time() - last_detection_time > loss_duration:
                    break
            elif time.time() - last_detection_time > loss_duration:
                break
            cv2.imshow("Depth Map", depth_3channel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break
                # Show images
        d_reset = 1
        # Stop drone movement

        drone.send_rc_control(0, forward_speed, 0, 0)
        time.sleep(time_forwrd)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
            exit_flag.set()
            break


def main():
    drone = initialize_drone()
    depth_model, transform, device = initialize_depth_anything()
    yolo_model = YOLO("best.pt").to(device)
    confidence = 0.7
    drone.takeoff()
    drone.move_up(50)
    yaw_PID = cvzone.PID([0.25, 0, 0.13], INPUT_WIDTH // 2)
    capture_thread = threading.Thread(target=capture_frames, args=(drone,))
    process_thread = threading.Thread(target=process_frames,
                                      args=(drone, depth_model, transform, device, yaw_PID, yolo_model, confidence))
    map_thread = threading.Thread(target=drawMap, args=())

    capture_thread.start()
    process_thread.start()
    map_thread.start()

    map_thread.join()
    capture_thread.join()
    process_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
