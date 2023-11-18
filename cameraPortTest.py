import cv2

# Get the list of available camera ports
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
    cap.release()

# Create windows for each available camera
for port in available_cameras:
    cap = cv2.VideoCapture(port)
    window_title = f"Camera Port {port}"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to a smaller size
        frame = cv2.resize(frame, (320, 240))

        # Display the frame in a window with the camera port title
        cv2.imshow(window_title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_title)

cv2.destroyAllWindows()
