import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))

import pickle
import face_recognition

# def initialize_target_user_detection():
#     """
#     Initializes the face recognition.
#     """
#     file = open('TargetDetection/EncodeFile.p', 'rb')
#     encodeListKnownWithTargets = pickle.load(file)
#     file.close()
#     return encodeListKnownWithTargets


# def find_user(drone, encodeListKnown, targets):
#     global img
#     while not exit_flag.is_set():  # Check for exit flag to stop the loop
#         frame_ready.wait()  # Wait until a new frame is ready for processing
#         frame_ready.clear()  # Clear the event to reset it for the next frame
#         img_resized = cv2.resize(img, (480, 480))
#         # Face recognition
#         faceCurFrame = face_recognition.face_locations(img_resized)
#         encodeCurFrame = face_recognition.face_encodings(img_resized, faceCurFrame)
#
#         for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
#             matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#             faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#             best_match_index = np.argmin(faceDis)
#             if matches[best_match_index]:
#                 name = targets[best_match_index]
#                 print(f"Detected: {name}")
#                 top, right, bottom, left = faceLoc
#                 cv2.rectangle(img_resized, (left, top), (right, bottom), (0, 255, 0), 2)
#                 cv2.putText(img_resized, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# encodeListKnown, targets = initialize_target_user_detection()