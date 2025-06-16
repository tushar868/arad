from ultralytics import YOLO

# Load YOLOv8n (you can change to yolov8s.pt or yolov8m.pt)
model = YOLO("yolov8n.pt")

# Train the model using your synthetic dataset
model.train(
    data="billboard_dataset/data.yaml",  # Replace with actual path
    epochs=50,
    imgsz=640
)
