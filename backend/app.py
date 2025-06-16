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

# Load Twitter client
BEARER_TOKEN = os.getenv("AAAAAAAAAAAAAAAAAAAAAMSj2QEAAAAAxddYYmf4%2BmgbxC%2BnRfHNq6jcP64%3DOaHe7oFVykBE9pj0EVEMyt15FORFay18rrrKEzngbTjOsOiMLy")
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Load YOLOv8 model
model = YOLO("runs/train/weights/best.pt")  # Adjust the path if needed

@app.route('/verify')
def verify():
    username = request.args.get('username', '').replace('@', '').strip()
    if not username:
        return jsonify({"success": False, "error": "Username missing"})

    try:
        user = client.get_user(username=username)
        tweets = client.get_users_tweets(id=user.data.id, max_results=5)

        for tweet in tweets.data:
            if '#AdReveal' in tweet.text:
                return jsonify({"success": True})

        return jsonify({"success": False})

    except Exception as e:
        print(f"Error verifying tweet: {e}")
        return jsonify({"success": False, "error": str(e)})

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

        # Run YOLOv8 detection
        results = model.predict(img, conf=0.25)[0]

        if hasattr(results, 'masks') and results.masks is not None:
            mask = results.masks.data[0].cpu().numpy()
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return jsonify({'error': 'No billboard contour found'}), 404

            epsilon = 0.02 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)

            if len(approx) != 4:
                return jsonify({'error': 'Billboard not rectangular'}), 422

            corners = [{'x': int(pt[0][0]), 'y': int(pt[0][1])} for pt in approx]
        else:
            if len(results.boxes) == 0:
                return jsonify({'error': 'No billboard detected'}), 404

            box = results.boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            corners = [
                {'x': x1, 'y': y1},
                {'x': x2, 'y': y1},
                {'x': x2, 'y': y2},
                {'x': x1, 'y': y2}
            ]

        return jsonify({'corners': corners})

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': 'Detection failed'}), 500

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
        print(f"Claim error: {e}")
        return jsonify({"error": "Failed to claim"}), 500

if __name__ == '__main__':
    app.run(debug=True)
