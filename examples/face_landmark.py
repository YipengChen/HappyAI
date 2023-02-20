import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_landmark import FaceLandmark

cap = cv2.VideoCapture(0)
face_landmark = FaceLandmark()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image[:, 80:-80, :], 1)
    results = face_landmark.inference(image)
    image = face_landmark.draw(image, results)
    
    cv2.imshow('Hand Detection', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
