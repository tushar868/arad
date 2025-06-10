from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import tweepy
import os
import json
import base64
import cv2
import numpy as np

app = Flask(__name__)

# === YOLOv8 Model ===
model = YOLO("runs/billboard_train/weights/best.pt")

def readb64(uri):
    """Decode base64 image to OpenCV format"""
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/detect', methods=['POST'])
def detect_billboard():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"success": False, "message": "No image data provided."}), 400

    image = readb64(data['image'])
    results = model.predict(source=image, conf=0.5, verbose=False)
    boxes = results[0].boxes

    if not boxes:
        return jsonify({"success": True, "detections": 0, "boxes": []})

    detected_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detected_boxes.append({
            "x": int(x1), "y": int(y1),
            "width": int(x2 - x1), "height": int(y2 - y1)
        })

    return jsonify({
        "success": True,
        "detections": len(boxes),
        "boxes": detected_boxes
    })


# === Twitter Verification ===
API_KEY = "GgyDJk7ENKfBwmxCJ8sUO5Lp5"
API_SECRET = "93RFPcTeKjkx25b7Bigag5jBkJjXjlGXfCaPX5lRky7ZZApF6R"
ACCESS_TOKEN = "1786050411234770945-MaSBI5D5CKES7y3btoOL0JNb4ZyTdM"
ACCESS_SECRET = "vHN8czcuMhqweNzC9e9LT1wD2t5hskQZgudznMsOBpcAL"

auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

COUPON_FILE = "coupons.json"
if not os.path.exists(COUPON_FILE):
    with open(COUPON_FILE, "w") as f:
        json.dump({f"AMZ-{i:03}": {"claimed": False} for i in range(10)}, f)

@app.route('/verify-tweet', methods=['POST'])
def verify_tweet():
    data = request.get_json()
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "message": "Username required"}), 400

    try:
        tweets = api.user_timeline(screen_name=username, count=10, tweet_mode="extended")
        for tweet in tweets:
            if "#AdReveal" in tweet.full_text:
                with open(COUPON_FILE, "r+") as f:
                    coupons = json.load(f)
                    for code, info in coupons.items():
                        if not info["claimed"]:
                            info["claimed"] = True
                            f.seek(0)
                            json.dump(coupons, f, indent=2)
                            f.truncate()
                            return jsonify({"success": True, "coupon": code})
                return jsonify({"success": False, "message": "All coupons claimed"})
        return jsonify({"success": False, "message": "Tweet not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# === Serve HTML Files ===
@app.route('/')
@app.route('/home.html')
def serve_home():
    return send_from_directory('.', 'home.html')

@app.route('/camera.html')
def serve_camera():
    return send_from_directory('.', 'camera.html')

@app.route('/voucher.html')
def serve_voucher():
    return send_from_directory('.', 'voucher.html')


# === Serve Asset Files (images/videos/scripts) ===
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
