Backend for Curtain Ad Detection Project

Endpoints:
/detect - Accepts a POST request with base64 image and returns YOLOv8m detection result.

Dependencies:
- Flask
- ultralytics
- opencv-python
- numpy
- Pillow

Model:
- Place 'best_yolov8m.pt' inside weights/ folder before running.

To run:
$ pip install -r requirements.txt
$ python app.py
