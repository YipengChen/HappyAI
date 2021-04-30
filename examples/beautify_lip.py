import cv2
import sys
sys.path.append("..")
from base_ai.face_ai import FaceMesh
import numpy as np

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    upper_lip_location, lower_lip_location = face_mesh.get_lip_location(results, image.shape)

    if upper_lip_location is not None and lower_lip_location is not None:
        lip_image = cv2.fillPoly(image.copy(), [upper_lip_location], (0, 0, 255))
        lip_image = cv2.fillPoly(lip_image, [lower_lip_location], (0, 0, 255))

    image = cv2.addWeighted(image, 0.8, lip_image, 0.2, gamma=0)
    
    cv2.imshow('Beautify Lip', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
