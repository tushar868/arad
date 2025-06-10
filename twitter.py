from flask import Flask, request, jsonify
import tweepy
import json
import os

app = Flask(__name__)

# Twitter API credentials
API_KEY = "GgyDJk7ENKfBwmxCJ8sUO5Lp5"
API_SECRET = "93RFPcTeKjkx25b7Bigag5jBkJjXjlGXfCaPX5lRky7ZZApF6R"
ACCESS_TOKEN = "1786050411234770945-MaSBI5D5CKES7y3btoOL0JNb4ZyTdM"
ACCESS_SECRET = "vHN8czcuMhqweNzC9e9LT1wD2t5hskQZgudznMsOBpcAL"

# Set up Twitter client
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

COUPON_FILE = "coupons.json"
if not os.path.exists(COUPON_FILE):
    with open(COUPON_FILE, "w") as f:
        json.dump({f"AMZ-{i:03}": {"claimed": False} for i in range(10)}, f)

@app.route('/verify-tweet', methods=['POST'])
def verify():
    data = request.json
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "message": "Username is required"}), 400

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)