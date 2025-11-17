from ultralytics import YOLO
import cv2
import os

def load_model(model_path="yolov8n.pt"):
    """Load YOLOv8 model."""
    try:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Downloading YOLOv8n model...")
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_objects(model, frame, conf=0.5):
    """Run object detection on a frame."""
    results = model.predict(frame, conf=conf, verbose=False)
    return results

"""def draw_boxes(frame, results):
    """"""Draw bounding boxes on detected objects.""""""
    for r in results:
        boxes = r.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame"""

def draw_boxes(frame, results):
    """Draw bounding boxes with class names and confidence."""
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{r.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
