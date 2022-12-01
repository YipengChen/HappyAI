# https://learnopencv.com/create-snapchat-instagram-filters-using-mediapipe/

import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh
from base_ai.image import calculate_delaunay_triangles_index, warp_triangle
import numpy as np
import csv
import time


def find_convex_hull_points(points):
    hull_points_index = cv2.convexHull(points[None, :, 1:3], clockwise=False, returnPoints=False)
    add_points_index = [
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,  # Outer lips
        60, 61, 62, 63, 64, 65, 66, 67,  # Inner lips
        27, 28, 29, 30, 31, 32, 33, 34, 35,  # Nose
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,  # Eyes
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26  # Eyebrows
    ]
    hull_points_index = np.array(list(hull_points_index.squeeze()) + add_points_index)
    hull_points = points[hull_points_index, 1:3]
    return hull_points, hull_points_index


def load_annotation_file(annotation_file):
    points = []
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            points.append([int(row[0]), int(row[1]), int(row[2])])
    points = np.array(points, np.float32)
    return points


def load_filter(image_path, annotation_path, has_alpha=True):

    filter_dict = {}
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    filter_dict['image'] = image[:, :, 0:3]
    filter_dict['image_alpha'] = cv2.merge((image[:, :, 3], image[:, :, 3], image[:, :, 3]))
    filter_dict['points'] = load_annotation_file(annotation_path)

    # Find convex hull for delaunay triangulation using the landmark points
    filter_dict['hull_points'], filter_dict['hull_points_index'] = find_convex_hull_points(filter_dict['points'])

    # Find Delaunay triangulation for convex hull points
    image_size = filter_dict['image'].shape
    rect = (0, 0, image_size[1], image_size[0])
    filter_dict['triangle_indexs'] = calculate_delaunay_triangles_index(rect, filter_dict['hull_points'])

    return filter_dict


cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()
ar_filter = load_filter('./images/anime.png', './images/anime_annotations.csv')

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    ar_points = face_mesh.get_ar_location(results, image.shape)

    draw_image = image.copy()
    if ar_points is not None:
        vaild_mask_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        for triangle_index in ar_filter['triangle_indexs']:
            ar_triangle_points = ar_filter['hull_points'][triangle_index]
            image_triangle_points = ar_points[ar_filter['hull_points_index']][triangle_index]
            warp_triangle(ar_filter['image'], draw_image, ar_triangle_points, image_triangle_points)
            warp_triangle(ar_filter['image_alpha'], vaild_mask_image, ar_triangle_points, image_triangle_points)
        vaild_mask_image /= 255.0
        invert_vaild_mask_image = (1.0, 1.0, 1.0) - vaild_mask_image
        draw_image = np.uint8(draw_image * vaild_mask_image + image * invert_vaild_mask_image)
    
    cv2.imshow('Mask Face', draw_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
