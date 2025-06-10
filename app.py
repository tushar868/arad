from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)
model = YOLO("runs/billboard_train/weights/best.pt")

def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if 'image' not in data:
        return jsonify({"success": False, "message": "No image provided."}), 400

    image = readb64(data['image'])
    results = model.predict(source=image, conf=0.5, verbose=False)
    boxes = results[0].boxes

    return jsonify({
        "success": True,
        "detections": len(boxes),
        "boxes": [{"x": int(b.xyxy[0][0]), "y": int(b.xyxy[0][1])} for b in boxes]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
