import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.hand.hand_landmark import HandLandmark

cap = cv2.VideoCapture(0)
hand_detection = HandLandmark()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image[:, :, :], 1)
    results = hand_detection.inference(image)
    image = hand_detection.draw(image, results)
    
    cv2.imshow('Hand Detection', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
