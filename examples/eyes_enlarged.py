import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    
    enlarged_image = face_mesh.eyes_enlarged(results, image, enlarge_factor=1)

    cv2.imshow('Eyes Enlarged', enlarged_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
