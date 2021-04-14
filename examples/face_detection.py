import cv2
import sys
sys.path.append("..")
from base_ai.face_ai import FaceDetection

cap = cv2.VideoCapture(0)
face_detection = FaceDetection()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_detection.inference(image)
    image = face_detection.draw(image, results)
    
    cv2.imshow('Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
