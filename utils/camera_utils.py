import cv2

def initialize_webcam(camera_index=0):
    """Initialize and return the webcam capture object."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise Exception("Could not open webcam")
    return cap

def get_frame_webcam(cap):
    """Read a frame from the webcam."""
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to grab frame from webcam")
    return frame

def test_camera(camera_index=0):
    """Test if the webcam can open."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Camera {camera_index} not available.")
        return False
    cap.release()
    return True
