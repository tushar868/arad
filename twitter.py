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

# Coupon file
COUPON_FILE = "coupons.json"

@app.route('/verify-tweet', methods=['POST'])
def verify_tweet():
    data = request.json
    username = data.get("username", "").strip()
    keyword = data.get("keyword", "#AdReveal")

    if not username:
        return jsonify({"success": False, "message": "Username is required."}), 400

    try:
        # Get latest tweets from user
        tweets = api.user_timeline(screen_name=username, count=10, tweet_mode="extended")

        # Check for valid tweet
        for tweet in tweets:
            if keyword.lower() in tweet.full_text.lower():
                with open(COUPON_FILE, "r+") as f:
                    coupons = json.load(f)

                    for code, info in coupons.items():
                        if not info["claimed"]:
                            coupons[code]["claimed"] = True
                            coupons[code]["user"] = username
                            coupons[code]["tweet"] = tweet.full_text

                            f.seek(0)
                            json.dump(coupons, f, indent=2)
                            f.truncate()

                            return jsonify({"success": True, "coupon": code}), 200

                    return jsonify({"success": False, "message": "All coupons have been claimed."}), 200

        return jsonify({"success": False, "message": "No valid tweet found."}), 200

    except tweepy.TweepyException as e:
        return jsonify({"success": False, "error": f"Tweepy error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
