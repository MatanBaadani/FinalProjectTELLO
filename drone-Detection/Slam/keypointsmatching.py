import cv2

# Initialize the ORB detector with 1000 features
orb = cv2.ORB_create(nfeatures=1000)

# Initialize the BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Open the webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, previous_frame = cap.read()
if not ret:
    print("Failed to capture image")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    # Read the current frame
    ret, current_frame = cap.read()
    if not ret:
        break

    # Convert frames to grayscale
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for both frames
    keypoints1, descriptors1 = orb.detectAndCompute(gray_previous, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_current, None)

    # Match descriptors between the two frames
    if descriptors1 is not None and descriptors2 is not None:
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw the top 10 matches
        matched_img = cv2.drawMatches(previous_frame, keypoints1, current_frame, keypoints2, matches[:10], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the matched image
        cv2.imshow("Matches", matched_img)

    # Update the previous frame
    previous_frame = current_frame.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
