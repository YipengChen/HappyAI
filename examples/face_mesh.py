import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face_ai import FaceMesh

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    image = face_mesh.draw(image, results)
    
    cv2.imshow('Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
