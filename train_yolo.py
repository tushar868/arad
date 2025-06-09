from ultralytics import YOLO

# Load the YOLOv8m model (recommended for accuracy)
model = YOLO("yolov8m.pt")

# Train the model using your labeled billboard dataset
model.train(
    data="billboard-dataset/data.yaml",  # path to your YAML file
    epochs=50,                            # number of training epochs
    imgsz=640,                            # input image size
    batch=8,                              # batch size (adjust based on GPU)
    project="runs",                       # folder to save outputs
    name="billboard_train",               # subfolder name
    save=True                             # ensure weights are saved
)
