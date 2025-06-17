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
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAMSj2QEAAAAAxddYYmf4%2BmgbxC%2BnRfHNq6jcP64%3DOaHe7oFVykBE9pj0EVEMyt15FORFay18rrrKEzngbTjOsOiMLy"  # Replace with your actual token for local testing
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# === Load YOLOv8 Model ===
model = YOLO("runs/detect/train/weights/best.pt")  # Adjust path to your trained weights

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

        # === Segmentation masks (preferred) ===
        if hasattr(results, 'masks') and results.masks is not None:
            mask = results.masks.data[0].cpu().numpy()
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return jsonify({'error': 'No contour found'}), 404

            approx = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)
            if len(approx) != 4:
                return jsonify({'error': 'Billboard not rectangular'}), 422

            corners = [{'x': int(pt[0][0]), 'y': int(pt[0][1])} for pt in approx]

        # === Fallback to bounding box ===
        elif results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            corners = [
                {'x': x1, 'y': y1},
                {'x': x2, 'y': y1},
                {'x': x2, 'y': y2},
                {'x': x1, 'y': y2}
            ]
        else:
            return jsonify({'error': 'No billboard detected'}), 404

        return jsonify({'corners': corners})

    except Exception as e:
        print(f"[Detection Error] {e}")
        return jsonify({'error': 'Detection failed'}), 500

# === Fallback GET for /detect (method not allowed) ===
@app.route('/detect', methods=['GET'])
def detect_get_method_not_allowed():
    return jsonify({"error": "Method Not Allowed. Use POST"}), 405

# === ROUTE: Twitter Verification ===
@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        username = data.get('username', '').replace('@', '').strip()

        if not username:
            return jsonify({"verified": False, "error": "Username missing"}), 400

        user = client.get_user(username=username)
        if not user.data or not user.data.id:
            return jsonify({"verified": False, "error": "User not found"}), 404

        tweets = client.get_users_tweets(id=user.data.id, max_results=5)

        for tweet in tweets.data or []:
            if '#AdReveal' in tweet.text:
                return jsonify({"verified": True})

        return jsonify({"verified": False})

    except Exception as e:
        print(f"[Twitter Verify Error] {e}")
        return jsonify({"verified": False, "error": str(e)}), 500

# === ROUTE: Claim Coupon ===
@app.route('/claim')
def claim_coupon():
    try:
        with open('coupons.json', 'r') as f:
            data = json.load(f)

        for coupon in data['coupons']:
            if not coupon.get('claimed'):
                coupon['claimed'] = True
                with open('coupons.json', 'w') as fw:
                    json.dump(data, fw, indent=2)
                return jsonify({"coupon": coupon['code']})

        return jsonify({"error": "No coupons left"}), 404

    except Exception as e:
        print(f"[Claim Error] {e}")
        return jsonify({"error": "Failed to claim"}), 500

# === MAIN ===
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
