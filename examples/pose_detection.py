import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.pose_ai import PoseDetection

cap = cv2.VideoCapture(0)
pose_detection = PoseDetection()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = pose_detection.inference(image)
    image = pose_detection.draw(image, results)
    
    cv2.imshow('Pose Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
