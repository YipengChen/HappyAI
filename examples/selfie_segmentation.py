import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.segmentation_ai import SelfieSegmentation

cap = cv2.VideoCapture(0)
selfie_segmentation = SelfieSegmentation()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = selfie_segmentation.inference(image)
    image = selfie_segmentation.draw(image, results)
    
    cv2.imshow('Pose Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
