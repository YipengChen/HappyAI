import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh
import numpy as np

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()

# https://learnopencv.com/create-snapchat-instagram-filters-using-mediapipe/

mask_image = cv2.imread('./images/surgical_blue.png', cv2.IMREAD_UNCHANGED)
#mask_six_points_location = np.array([[39, 92], [311, 18], [586, 85], [593, 302], [304, 503], [27, 299]])
mask_six_points_location = np.array([[139, 47], [311, 18], [497, 50], [443, 302], [304, 503], [164, 302]])

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    face_six_points_location = face_mesh.get_mask_location(results, image.shape)

    if face_six_points_location is not None:        
        M, _ = cv2.findHomography(mask_six_points_location, face_six_points_location)
        transformed_mask_image = cv2.warpPerspective(mask_image.copy(), M, (image.shape[1], image.shape[0]))
        vaild_mask = np.squeeze(transformed_mask_image[:, :, 3])
        image[vaild_mask >= 128] = transformed_mask_image[vaild_mask >= 128][:,:3]
    
    cv2.imshow('Mask Face', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
