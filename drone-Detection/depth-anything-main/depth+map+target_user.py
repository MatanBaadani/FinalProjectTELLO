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

# Constants for configuration
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
# Global variables0
img = None
frame_lock = threading.Lock()
frame_ready = threading.Event()
exit_flag = threading.Event()
found_user = 0
found_Target = 0
route = np.zeros((1000, 1000, 3), np.uint8)
aSpeed = 12  # Angular Speed Degrees/s  25
interval = 0.1
dInterval = 0.5
start_point = [500, 800]
d = 0
a = 270
d_reset = 0
x = 500
y = 800
counter_of_movements = 0
optimal_route = []
ready_to_end = 0


def initialize_target_user_detection():
    """
    Initializes the face recognition.
    """
    file = open('TargetDetection/EncodeFile.p', 'rb')
    encodeListKnownWithTargets = pickle.load(file)
    file.close()
    return encodeListKnownWithTargets


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
                    print("found target")
                if name == "User" and found_user == 0:
                    found_user = 1
            else:
                top, right, bottom, left = faceLoc
                cv2.rectangle(img_resized, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.line(img_resized, (left, bottom), (right, top), (0, 0, 255), 2)
                cv2.line(img_resized, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(img_resized, "NOT_TARGET", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        # Display the depth map and face recognition
        cv2.imshow("Detected", img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag.set()
            break


def calculate_angle(point1, point2):
    """
    Calculate the angle in degrees between two points (x1, y1) and (x2, y2).

    Args:
        point1 (tuple): (x1, y1) coordinates of the first point.
        point2 (tuple): (x2, y2) coordinates of the second point.

    Returns:
        float: Angle in degrees from point1 to point2.
    """
    x1, y1 = point1
    x2, y2 = point2
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def optimal_path(coordinates):
    seen = set()
    path = []

    for point in coordinates:
        # Convert the point to a tuple to use it in a set
        point_tuple = tuple(point)

        if point_tuple in seen:
            # A cycle has been detected; skip this and all points until we get out of the cycle
            while path and tuple(path[-1]) != point_tuple:
                path.pop()  # Remove the cycle points
        else:
            seen.add(point_tuple)
            path.append(point)

    return path


def drawMap():
    global route, a, x, y, d, img, d_reset, start_point, found_user, found_Target, optimal_route
    # List to store joints (start points)
    joints = [start_point[:]]
    while not exit_flag.is_set():
        frame_ready.wait()  # Wait until a new frame is ready for processing
        frame_ready.clear()  # Clear the event to reset it for the next frame
        # Stage 1: Rotate 360 degrees and detect doors
        with frame_lock:  # Acquire the lock to safely read the shared 'img' variable
            if img is not None and d_reset != 0 and found_Target == 0:
                scale_factor = 0.85 if d > 130 else 0.15
                # Update position
                x += int(d * math.cos(math.radians(a)))
                y += int(d * math.sin(math.radians(a)))
                print(a, d, x, y)
                d, d_reset = 0, 0
                door_x = int((x - start_point[0]) * scale_factor + start_point[0])
                door_y = int((y - start_point[1]) * scale_factor + start_point[1])
                cv2.circle(route, tuple(start_point), 4, (0, 0, 255), cv2.FILLED)
                cv2.rectangle(route, (door_x - 3, door_y - 6), (door_x + 3, door_y + 6), (0, 255, 255), 2)
                cv2.line(route, tuple(start_point), (x, y), (0, 0, 255), 2)
                # Draw a rectangle around the detected door
                rect_top_left = (min(start_point[0], door_x) - 3, min(start_point[1], door_y) - 6)
                rect_bottom_right = (max(start_point[0], door_x) + 3, max(start_point[1], door_y) + 6)
                cv2.rectangle(route, rect_top_left, rect_bottom_right, (255, 255, 0), 1)
                if found_user == 1 and counter_of_movements == 0:
                    # Draw a circle for the head
                    cv2.circle(route, (start_point[0], start_point[1] - 10), 6, (0, 255, 0), -1)
                    # Draw a line for the body
                    cv2.line(route, (start_point[0], start_point[1]), (start_point[0], start_point[1] + 20),
                             (0, 255, 0), 2)
                    # Draw arms (simple lines)
                    cv2.line(route, (start_point[0] - 10, start_point[1] + 10),
                             (start_point[0] + 10, start_point[1] + 10), (0, 255, 0), 2)
                start_point[:] = [x, y]
                joints.append(start_point[:])
            if found_Target == 1 and ready_to_end == 1:
                # Compute target position (50 pixels in direction 'a')
                target_x = int(start_point[0] + 50 * math.cos(math.radians(a)))
                target_y = int(start_point[1] + 50 * math.sin(math.radians(a)))
                cv2.line(route, tuple(start_point), (target_x, target_y), (0, 0, 255), 2)
                cv2.circle(route, (target_x, target_y), 6, (255, 0, 255), 1)
                cv2.line(route, (target_x - 4, target_y), (target_x + 4, target_y), (255, 0, 255),
                         1)
                cv2.line(route, (target_x, target_y - 4), (target_x, target_y + 4), (255, 0, 255), 1)  # Vertical cross
                joints.append([target_x, target_y])
                optimal_route = optimal_path(joints)
                print(joints)
                print(optimal_route)
                if len(optimal_route) > 1:
                    for i in range(len(optimal_route) - 1):
                        cv2.line(route, tuple(optimal_route[i]), tuple(optimal_route[i + 1]), (255, 0, 0), 2)
                    np.savez("data.npz", array=route, list=np.array(optimal_route, dtype=object))
                exit_flag.set()
            cv2.imshow("Map", route)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break
    while True:

        cv2.imshow("Map", route)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
    while True:
        frame = drone.get_frame_read().frame
        with frame_lock:
            img = frame.copy()
            frame_ready.set()
        time.sleep(interval)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'q' key is pressed to exit the loop
            exit_flag.set()
            break


def detect_doors_yolo(model, depth_img, confidence):
    """
    Detects doors using the YOLO model.
    """
    results = model(depth_img, stream=False, verbose=False)
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
                object_list = [{
                    "bbox": (x1, y1, w, h),
                    "center": center,
                    "conf": conf,
                    "class": "open_door",
                }]
                cvzone.cornerRect(depth_img, (x1, y1, w, h))
                cvzone.putTextRect(depth_img, "open_door" f'{conf}', (max(0, x1), max(35, y1)), scale=1,
                                   thickness=1)

    return object_list


def process_frame_for_depth_optimized(model, transform, DEVICE):
    """
    Optimized function to process an image for depth estimation.
    """
    global img
    input_size = (INPUT_WIDTH, INPUT_HEIGHT)
    img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img_transformed = transform({'image': img_resized})['image']
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(DEVICE, non_blocking=True)

    with torch.no_grad():
        depth = model(img_tensor)

    depth_resized = F.interpolate(depth.unsqueeze(0), size=input_size, mode='bilinear', align_corners=False)[0, 0]
    depth_resized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
    depth_final = torch.clamp(depth_resized * 255, 0, 255).to(torch.uint8).cpu().numpy()
    depth_img_color = cv2.cvtColor(depth_final, cv2.COLOR_GRAY2BGR)

    return depth_img_color


def process_frames(drone, model, transform, DEVICE, yaw_PID, model_obstacle, confidence, rotation_time=30,
                   rotation_speed=25, consistency_threshold=1.6, detection_cooldown=3.0, pid_duration=7,
                   forward_speed=25, loss_duration=1):
    global img, a, d, d_reset, found_Target, found_user, counter_of_movements, ready_to_end

    while not exit_flag.is_set() and found_Target == 0 and (counter_of_movements == 0 or (
            counter_of_movements != 0 and found_user != 0)):  # Check for exit flag to stop the loop # Check for exit flag to stop the loop
        start_time = time.time()
        detection_times = []
        detection_start_time = None
        last_detection_time = None

        # Stage 1: Rotate 360 degrees and detect doors
        while time.time() - start_time < rotation_time and found_Target == 0 and counter_of_movements != 3:
            frame_ready.wait()  # Wait until a new frame is ready for processing
            frame_ready.clear()  # Clear the event to reset it for the next frame

            with frame_lock:  # Acquire the lock to safely read the shared 'img' variable
                if img is not None:  # Check if 'img' is not None to avoid errors
                    depth_img = process_frame_for_depth_optimized(model, transform, DEVICE)
            drone.send_rc_control(0, 0, 0, -rotation_speed)
            objects = detect_doors_yolo(model_obstacle, depth_img, confidence)

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
            cv2.imshow("Depth Map", depth_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break
        if found_Target == 0:
            if detection_times:
                first_door_time = detection_times[0] * 0.8
            print(f"Rotating to the detected door at {first_door_time} seconds...")
            drone.send_rc_control(0, 0, 0, -rotation_speed)
            a -= aSpeed * first_door_time
            time.sleep(first_door_time)
            drone.send_rc_control(0, 0, 0, 0)
        else:
            time_to_target = time.time() - start_time
            a -= aSpeed * time_to_target
            drone.send_rc_control(0, 0, 0, -rotation_speed)
            time.sleep(2)
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(2)
            ready_to_end = 1
            break
        # Stage 2: PID control to center the drone on the first detected door
        if found_Target == 0:
            pid_start_time = time.time()
        while time.time() - pid_start_time < pid_duration and found_Target == 0:
            depth_img = process_frame_for_depth_optimized(model, transform, DEVICE)
            objects = detect_doors_yolo(model_obstacle, depth_img, confidence)
            if objects:
                detected_door = objects[0]
                x_center_0, y_center = detected_door["center"]
                x_val = int(yaw_PID.update(x_center_0))
                drone.send_rc_control(0, 0, 0, x_val)
            else:
                drone.send_rc_control(0, 0, 0, 0)

            # Show images
            cv2.imshow("Depth Map", depth_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break

        # Stage 3: Move forward while adjusting yaw using PID
        while found_Target == 0:
            depth_img = process_frame_for_depth_optimized(model, transform, DEVICE)
            objects = detect_doors_yolo(model_obstacle, depth_img, confidence)
            if objects:
                closest_object = min(objects, key=lambda obj: abs(obj["bbox"][0] - x_center_0))
                x_center, y_center = closest_object["center"]
                if abs(x_center - x_center_0) < 30:
                    x_val = int(yaw_PID.update(x_center))
                    drone.send_rc_control(0, forward_speed, 0, x_val)
                    d += dInterval
                    last_detection_time = time.time()
                    x_center_0 = x_center
                elif time.time() - last_detection_time > loss_duration:
                    x_center_0 = x_center
                    break
            elif time.time() - last_detection_time > loss_duration:
                break
            cv2.imshow("Depth Map", depth_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag.set()
                break
        # Stop drone movement
        if found_Target == 0:
            drone.send_rc_control(0, forward_speed, 0, 0)
            time_forward = 2 if d > 130 else 5
            d += time_forward * 5
            d_reset = 1
            time.sleep(time_forward)
        else:
            d_reset = 1
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(2)
            break
        counter_of_movements += 1


def return_home(drone, model, transform, DEVICE, yaw_PID, model_obstacle, confidence,
                rotation_speed=25, pid_duration=8, forward_speed=20, loss_duration=1):
    global found_user, optimal_route, route, d  # Declare 'img' as global to access it within this function

    rotating_angles = []
    for i in range(len(optimal_route) - 2, 0, -1):
        if i == len(optimal_route) - 2:
            currnet_angle = calculate_angle(optimal_route[i], optimal_route[i + 1])
        next_angle = calculate_angle(optimal_route[i], optimal_route[i - 1])
        rotating_angles.append(next_angle - currnet_angle)
        currnet_angle = next_angle
    for i in range(len(rotating_angles)):
        rotation_time = abs(rotating_angles[i]) / aSpeed
        print("rotating to estimated open door in the optimal route")
        if rotating_angles[i] < 0:
            drone.send_rc_control(0, 0, 0, -rotation_speed)
        else:
            drone.send_rc_control(0, 0, 0, rotation_speed)
        time.sleep(rotation_time)
        pid_start_time = time.time()
        while time.time() - pid_start_time < pid_duration:
            depth_img = process_frame_for_depth_optimized(model, transform, DEVICE)
            objects = detect_doors_yolo(model_obstacle, depth_img, confidence)
            if objects:
                detected_door = objects[0]
                x_center_0, y_center = detected_door["center"]
                x_val = int(yaw_PID.update(x_center_0))
                drone.send_rc_control(0, 0, 0, x_val)
            else:
                drone.send_rc_control(0, 0, 0, 0)

            # Show images
            cv2.imshow("Route", depth_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stage 3: Move forward while adjusting yaw using PID
        while True:
            depth_img = process_frame_for_depth_optimized(model, transform, DEVICE)
            objects = detect_doors_yolo(model_obstacle, depth_img, confidence)
            if objects:
                closest_object = min(objects, key=lambda obj: abs(obj["bbox"][0] - x_center_0))
                x_center, y_center = closest_object["center"]
                if abs(x_center - x_center_0) < 30:
                    x_val = int(yaw_PID.update(x_center))
                    drone.send_rc_control(0, forward_speed, 0, x_val)
                    d += dInterval
                    last_detection_time = time.time()
                    x_center_0 = x_center
                elif time.time() - last_detection_time > loss_duration:
                    x_center_0 = x_center
                    break
            elif time.time() - last_detection_time > loss_duration:
                break
            cv2.imshow("Depth Map", depth_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        drone.send_rc_control(0, forward_speed, 0, 0)
        time_forward = 2 if 130 > d > 50 and d != 0 else 5
        d += time_forward * 5
        time.sleep(time_forward)
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()


def main():
    drone = initialize_drone()
    depth_model, transform, device = initialize_depth_anything()
    yolo_model = YOLO("best.pt").to(device)
    encodeListKnown, targets = initialize_target_user_detection()
    confidence = 0.66
    drone.takeoff()
    time.sleep(2)
    yaw_PID = cvzone.PID([0.35, 0, 0.13], INPUT_WIDTH // 2)
    capture_thread = threading.Thread(target=capture_frames, args=(drone,))
    process_thread = threading.Thread(target=process_frames,
                                      args=(drone, depth_model, transform, device, yaw_PID, yolo_model, confidence))
    map_thread = threading.Thread(target=drawMap, args=())
    target_user_thread = threading.Thread(target=find_user_and_target, args=(encodeListKnown, targets))
    return_thread = threading.Thread(target=return_home,
                                     args=(drone, depth_model, transform, device, yaw_PID, yolo_model, confidence))

    capture_thread.start()
    process_thread.start()
    map_thread.start()
    target_user_thread.start()

    process_thread.join()
    target_user_thread.join()
    return_thread.start()
    return_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
