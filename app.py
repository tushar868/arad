from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load YOLOv8 model (use correct path to your trained weights)
model = YOLO("runs/billboard_train/weights/best.pt")  # change if path differs

def readb64(uri):
    """Decode base64 image into OpenCV format"""
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({"success": False, "message": "No image provided."}), 400

    # Decode the image
    image = readb64(data['image'])

    # Run detection using YOLOv8
    results = model.predict(source=image, conf=0.5, verbose=False)

    # Extract detected boxes
    boxes = results[0].boxes

    # Optional: You can return bbox coordinates like this
    detected_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detected_boxes.append({"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)})

    return jsonify({
        "success": True,
        "detections": len(boxes),
        "boxes": detected_boxes
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
