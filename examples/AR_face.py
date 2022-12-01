import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh
import numpy as np
import csv
import time

# https://learnopencv.com/create-snapchat-instagram-filters-using-mediapipe/

def load_annotation_file(annotation_file):
    points = []
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            points.append([int(row[0]), int(row[1]), int(row[2])])
    points = np.array(points, np.float32)
    return points


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


# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculate_delaunay_triangles_index(rect, points):

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangle_list = np.array(subdiv.getTriangleList())
    print(triangle_list)

    # Find the indices of triangles in the points array
    triangles_index = []
    points = points.tolist()
    for triangle in triangle_list:
        triangle = list(triangle)
        point_1_index = points.index(triangle[0:2])
        point_2_index = points.index(triangle[2:4])
        point_3_index = points.index(triangle[4:6])
        triangles_index.append([point_1_index, point_2_index, point_3_index])
    assert triangle_list.shape[0] == len(triangles_index)
    return triangles_index


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

# Warps and alpha blends triangular regions from image1 to image2
def warp_triangle(image1, image2, triangle1, triangle2):
    # Find bounding rectangle for each triangle, (x, y, width, height)
    rect1 = cv2.boundingRect(np.float32([triangle1]))
    rect2 = cv2.boundingRect(np.float32([triangle2]))
    rect2_vaild_x1, rect2_vaild_y1 = min(max(rect2[0], 0), image2.shape[1]), min(max(rect2[1], 0), image2.shape[0])
    rect2_vaild_x2, rect2_vaild_y2 = min(max(rect2[0]+rect2[2], 0), image2.shape[1]), min(max(rect2[1]+rect2[3], 0), image2.shape[0])
    rect2_width, rect2_height = rect2_vaild_x2-rect2_vaild_x1, rect2_vaild_y2-rect2_vaild_y1
    rect1_image = image1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]

    # Offset points by left top corner of the respective rectangles
    norm_triangle1 = []
    norm_triangle2 = []
    for i in range(0, 3):
        norm_triangle1.append(((triangle1[i][0] - rect1[0]), (triangle1[i][1] - rect1[1])))
        norm_triangle2.append(((triangle2[i][0] - rect2[0]), (triangle2[i][1] - rect2[1])))

    # Get mask by filling triangle
    rect2_vaild_mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(rect2_vaild_mask, np.int32(norm_triangle2), (1.0, 1.0, 1.0), 16, 0)
    rect2_vaild_mask = rect2_vaild_mask[rect2_vaild_y1-rect2[1]:rect2_vaild_y2-rect2[1], rect2_vaild_x1-rect2[0]:rect2_vaild_x2-rect2[0], :]
    invert_rect2_vaild_mask = (1.0, 1.0, 1.0) - rect2_vaild_mask

    # Apply affine transform
    warp_Mat = cv2.getAffineTransform(np.float32(norm_triangle1), np.float32(norm_triangle2))
    rect2_image = cv2.warpAffine(rect1_image, warp_Mat, (rect2[2], rect2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    rect2_image = rect2_image[rect2_vaild_y1-rect2[1]:rect2_vaild_y2-rect2[1], rect2_vaild_x1-rect2[0]:rect2_vaild_x2-rect2[0], :]
    rect2_image = rect2_image * rect2_vaild_mask
    rect2_image = rect2_image.astype(np.uint8)

    # Copy triangular region of the rectangular patch to the output image
    image2[rect2_vaild_y1:rect2_vaild_y2, rect2_vaild_x1:rect2_vaild_x2] = image2[rect2_vaild_y1:rect2_vaild_y2, rect2_vaild_x1:rect2_vaild_x2] * invert_rect2_vaild_mask
    image2[rect2_vaild_y1:rect2_vaild_y2, rect2_vaild_x1:rect2_vaild_x2] += rect2_image


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
