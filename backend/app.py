from flask import Flask, request, jsonify
from flask_cors import CORS
import tweepy
import os
import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# === Twitter API Setup ===
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not BEARER_TOKEN:
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMSj2QEAAAAAxddYYmf4%2BmgbxC%2BnRfHNq6jcP64%3DOaHe7oFVykBE9pj0EVEMyt15FORFay18rrrKEzngbTjOsOiMLy"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# === Load YOLOv8 Model ===
model = YOLO("runs/detect/train/weights/best.pt")

# === ROUTE: Detect billboard from base64 image ===
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'Image data missing'}), 400

        encoded_data = data.split(',')[1]
        img_data = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model.predict(img, conf=0.25)[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return jsonify({'error': 'No billboard detected'}), 404

        max_area = 0
        best_box = None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if area > max_area and is_center_priority(cx, cy, img.shape):
                max_area = area
                best_box = [x1, y1, x2, y2]

        if not best_box:
            return jsonify({'error': 'No central billboard detected'}), 404

        x1, y1, x2, y2 = best_box
        corners = [
            {'x': x1, 'y': y1},
            {'x': x2, 'y': y1},
            {'x': x2, 'y': y2},
            {'x': x1, 'y': y2},
        ]

        return jsonify({'corners': corners})

    except Exception as e:
        print(f"[Detection Error] {e}")
        return jsonify({'error': 'Detection failed'}), 500

def is_center_priority(cx, cy, shape):
    h, w = shape[:2]
    center_margin_x = w * 0.2
    center_margin_y = h * 0.2
    return (w/2 - center_margin_x < cx < w/2 + center_margin_x and
            h/2 - center_margin_y < cy < h/2 + center_margin_y)

@app.route('/detect', methods=['GET'])
def detect_get():
    return jsonify({'error': 'Use POST method'}), 405

@app.route('/verify', methods=['POST'])
def verify():
    try:
        username = request.json.get("username", "").replace("@", "").strip()
        if not username:
            return jsonify({"verified": False, "error": "Username required"}), 400

        user = client.get_user(username=username)
        if not user.data:
            return jsonify({"verified": False, "error": "User not found"}), 404

        tweets = client.get_users_tweets(id=user.data.id, max_results=5)
        for tweet in tweets.data or []:
            if "#AdReveal" in tweet.text:
                return jsonify({"verified": True})

        return jsonify({"verified": False})

    except Exception as e:
        print(f"[Twitter Verify Error] {e}")
        return jsonify({"verified": False, "error": str(e)}), 500

@app.route('/claim')
def claim():
    try:
        with open('coupons.json', 'r') as f:
            data = json.load(f)

        for coupon in data['coupons']:
            if not coupon.get('claimed'):
                coupon['claimed'] = True
                with open('coupons.json', 'w') as f2:
                    json.dump(data, f2, indent=2)
                return jsonify({"coupon": coupon['code']})

        return jsonify({"error": "No coupons left"}), 404

    except Exception as e:
        print(f"[Claim Error] {e}")
        return jsonify({"error": "Failed to claim"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
