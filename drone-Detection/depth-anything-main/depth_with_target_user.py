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
import pickle
import face_recognition

map = np.zeros((1000, 1000, 3), np.uint8)
fSpeed = 117 / 10  # Forward Speed in cm/s   20
aSpeed = 12  # Angular Speed Degrees/s  25
interval = 0.07
dInterval = fSpeed * interval
aInterval = aSpeed * interval
x, y = 500, 500
a = 0
yaw = 0
d=0




# Constants for configuration
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.80
# Global variables
img = None
frame_lock = threading.Lock()
frame_ready = threading.Event()
exit_flag = threading.Event()

found_user = 0
found_Target = 0


def initialize_drone():
    """
    Initializes the Tello drone and starts the video stream.
    """
    me = Tello()
    me.connect()
    print(me.get_battery())
    me.streamon()
    return me


def initialize_target_user_detection():
    """
    Initializes the face recognition.
    """
    file = open('TargetDetection/EncodeFile.p', 'rb')
    encodeListKnownWithTargets = pickle.load(file)
    file.close()
    return encodeListKnownWithTargets


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
    print(device)
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
        time.sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
            exit_flag.set()
            break


def find_user_and_target(encodeListKnown, targets):
    global img, found_Target, found_user
    while not exit_flag.is_set():  # Check for exit flag to stop the loop
        frame_ready.wait()  # Wait until a new frame is ready for processing
        frame_ready.clear()  # Clear the event to reset it for the next frame
        image_resized = cv2.resize(img, (240, 240))
        img_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        # Face recognition
        faceCurFrame = face_recognition.face_locations(img_resized)
        encodeCurFrame = face_recognition.face_encodings(img_resized, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            best_match_index = np.argmin(faceDis)
            if matches[best_match_index]:
                name = targets[best_match_index]
                print(f"Detected: {name}")
                top, right, bottom, left = faceLoc
                cv2.rectangle(img_resized, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_resized, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if name == "Elon_Musk" and found_Target == 0:
                    found_Target = 1
                if name == "User" and found_user == 0:
                    found_user = 1
        # Display the depth map and face recognition
        cv2.imshow("Detected", img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_doors_yolo(model, img, class_names, confidence):
    """
    Detects doors using the YOLO model.
    """
    results = model(img, stream=False, verbose=False)
    object_list = []
    area = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > confidence:
                class_name = class_names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                center = x1 + w // 2, y1 + h // 2
                object_list.append({"bbox": (x1, y1, w, h), "center": center, "conf": conf, "class": class_name})
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    return object_list, area


def process_frames(drone, model, transform, DEVICE, yaw_PID, model_obstacle, class_names, confidence, rotation_time=30,
                   rotation_speed=25, consistency_threshold=1.5, detection_cooldown=3.0, pid_duration=12,
                   forward_speed=20, loss_duration=1, time_forwrd=2.5):
    global img, found_Target, found_user, a,d  # Declare 'img' as global to access it within this function

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
        return detect_doors_yolo(model_obstacle, depth_3channel, class_names, confidence), depth_3channel

    counter_of_movements = 0
    while not exit_flag.is_set() and found_Target == 0 and ((counter_of_movements == 0 and found_user == 0) or (
            counter_of_movements != 0 and found_user != 0)):  # Check for exit flag to stop the loop
        start_time = time.time()

        detection_times = []
        detection_start_time = None
        last_detection_time = None

        # Stage 1: Rotate 360 degrees and detect doors
        while time.time() - start_time < rotation_time and found_Target == 0:
            frame_ready.wait()  # Wait until a new frame is ready for processing
            frame_ready.clear()  # Clear the event to reset it for the next frame

            with frame_lock:  # Acquire the lock to safely read the shared 'img' variable
                if img is not None:  # Check if 'img' is not None to avoid errors
                    depth_img = process_frame_for_depth_optimized()
            drone.send_rc_control(0, 0, 0, rotation_speed)
            objects_and_area, depth_3channel = detect_doors(depth_img)
            detected_doors = objects_and_area[0]

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
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        if found_Target == 0:
            if detection_times:
                first_door_time = detection_times[0] - 3
                print(f"Rotating to the first detected door at {first_door_time} seconds...")
                drone.send_rc_control(0, 0, 0, rotation_speed)
                a = aSpeed * first_door_time
                time.sleep(first_door_time)
                drone.send_rc_control(0, 0, 0, 0)

            # Stage 2: PID control to center the drone on the first detected door
            pid_start_time = time.time()
            last_detection_time = time.time()
        while time.time() - pid_start_time < pid_duration and found_Target == 0:
            depth_img = process_frame_for_depth_optimized()
            objects_and_area, depth_3channel = detect_doors(depth_img)
            detected_doors, area_0 = objects_and_area
            if detected_doors:
                x, y = detected_doors[0]["center"]
                x_val = int(yaw_PID.update(x))
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
        while found_Target == 0:
            depth_img = process_frame_for_depth_optimized()
            objects_and_area, depth_3channel = detect_doors(depth_img)
            detected_doors, area = objects_and_area
            if detected_doors and (area >= area_0 - 1000 or area <= area_0 + 2000):
                x, y = detected_doors[0]["center"]
                x_val = int(yaw_PID.update(x))
                drone.send_rc_control(0, forward_speed, 0, x_val)
                a+=270
                d=dInterval
                last_detection_time = time.time()
            elif time.time() - last_detection_time > loss_duration:
                break
            area_0 = area
            # Show images
            cv2.imshow("Depth Map", depth_3channel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break

        # Stop drone movement
        if found_Target == 0:
            drone.send_rc_control(0, forward_speed, 0, 0)
            time.sleep(time_forwrd)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
                exit_flag.set()
                break
        counter_of_movements += 1
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(10)
    drone.land()


def main():
    drone = initialize_drone()
    depth_model, transform, device = initialize_depth_anything()
    yolo_model = YOLO("best.pt").to(device)
    encodeListKnown, targets = initialize_target_user_detection()
    class_names = ["obstacle"]
    confidence = 0.65
    drone.takeoff()
    drone.move_up(50)
    yaw_PID = cvzone.PID([0.25, 0, 0.13], INPUT_WIDTH // 2)
    capture_thread = threading.Thread(target=capture_frames, args=(drone,))
    process_thread = threading.Thread(target=process_frames, args=(
        drone, depth_model, transform, device, yaw_PID, yolo_model, class_names, confidence))
    target_user_thread = threading.Thread(target=find_user_and_target, args=(encodeListKnown, targets))

    capture_thread.start()
    process_thread.start()
    target_user_thread.start()

    target_user_thread.join()
    capture_thread.join()
    process_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
