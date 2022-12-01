# https://learnopencv.com/create-snapchat-instagram-filters-using-mediapipe/

import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh
from base_ai.image import calculate_delaunay_triangles_index, warp_triangle, find_convex_hull_points
import numpy as np
import time

face_mesh = FaceMesh()

target_face_image = cv2.imread('./images/caixukun.png')
target_face = face_mesh.inference(target_face_image)
target_face_points = face_mesh.get_all_location(target_face, target_face_image.shape)
target_triangle_indexs = calculate_delaunay_triangles_index((0, 0, target_face_image.shape[1], target_face_image.shape[0]), target_face_points)

cap = cv2.VideoCapture(0)
while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    face_points = face_mesh.get_all_location(results, image.shape, has_z=True)

    draw_image = image.copy()
    if face_points is not None:
        hull_points = find_convex_hull_points(face_points[:, 0:2])
        vaild_mask_image = np.zeros((draw_image.shape), dtype=np.float32)
        cv2.fillConvexPoly(vaild_mask_image, np.int32(hull_points), (1.0, 1.0, 1.0), 16, 0)
        vaild_mask_image = cv2.erode(vaild_mask_image, np.ones((5, 5), dtype=np.uint8), iterations=1)
        vaild_mask_image = cv2.GaussianBlur(vaild_mask_image, (21, 21), 100, 100)
        invert_vaild_mask_image = (1.0, 1.0, 1.0) - vaild_mask_image

        triangle_pairs = []
        for triangle_index in target_triangle_indexs:
            target_triangle_points = target_face_points[triangle_index]
            image_triangle_points = face_points[:,0:2][triangle_index]
            image_triangle_z = face_points[:,2][triangle_index].mean()
            triangle_pairs.append([target_triangle_points, image_triangle_points, image_triangle_z])

        triangle_pairs.sort(key=lambda item: -item[2])
        
        for triangle_pair in triangle_pairs:
            warp_triangle(target_face_image, draw_image, triangle_pair[0], triangle_pair[1])
            
        draw_image = np.uint8(draw_image * vaild_mask_image + image * invert_vaild_mask_image)
    
    cv2.imshow('Mask Face', draw_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
