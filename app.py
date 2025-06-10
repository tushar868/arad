from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load your YOLOv8 model (adjust path to match yours)
model = YOLO("runs/billboard_train/weights/best.pt")

# Serve HTML files from root
@app.route("/")
def home():
    return send_from_directory(".", "home.html")

@app.route("/camera.html")
def camera():
    return send_from_directory(".", "camera.html")

@app.route("/voucher.html")
def voucher():
    return send_from_directory(".", "voucher.html")

# Serve assets (images, videos, icons, etc.)
@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory("assets", filename)

# YOLOv8 detection endpoint
@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({"success": False, "message": "No image provided"}), 400

    try:
        # Decode base64 image
        encoded_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLOv8 detection
        results = model.predict(source=img, conf=0.5, verbose=False)
        boxes = results[0].boxes

        # Return detection result
        detected_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detected_boxes.append({
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            })

        return jsonify({
            "success": True,
            "detections": len(boxes),
            "boxes": detected_boxes
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
