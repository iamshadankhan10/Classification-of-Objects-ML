from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # loads a pre-trained YOLOv8 Nano model
results = model.predict(source='https://ultralytics.com/images/bus.jpg', show=True)
