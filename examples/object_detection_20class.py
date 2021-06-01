import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.object_detection_ai import ObjectDetection_20class

cap = cv2.VideoCapture(0)
object_detection = ObjectDetection_20class()

while cap.isOpened():

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = object_detection.inference(image)
    image = object_detection.draw(image, results, confidence_threshold=0.8)
    
    cv2.imshow("ObjectDetection", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()