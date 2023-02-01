import cv2
import numpy as np


# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculate_delaunay_triangles_index(rect, points):

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangle_list = np.array(subdiv.getTriangleList())

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


# Warps and alpha blends triangular regions from image1 to image2
def warp_triangle(image1, image2, triangle1, triangle2):
    # Find bounding rectangle for each triangle, (x, y, width, height)
    rect1 = cv2.boundingRect(np.float32([triangle1]))
    rect1_image = image1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]

    if rect1_image.shape[0] == 0 or rect1_image.shape[1] == 0:
        return

    rect2 = cv2.boundingRect(np.float32([triangle2]))
    rect2_vaild_x1, rect2_vaild_y1 = min(max(rect2[0], 0), image2.shape[1]), min(max(rect2[1], 0), image2.shape[0])
    rect2_vaild_x2, rect2_vaild_y2 = min(max(rect2[0]+rect2[2], 0), image2.shape[1]), min(max(rect2[1]+rect2[3], 0), image2.shape[0])
    rect2_width, rect2_height = rect2_vaild_x2-rect2_vaild_x1, rect2_vaild_y2-rect2_vaild_y1

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


def find_convex_hull_points(points):
    hull_points = cv2.convexHull(np.float32(points[None, :, :]), clockwise=False, returnPoints=True)
    return hull_points.squeeze()


def overlap_image(raw_image, target_image, location):
    x, y, w, h = location
    raw_image_height, raw_image_width = raw_image.shape[0], raw_image.shape[1]
    target_image = cv2.resize(target_image.copy(), (h, w))
    raw_left_up = min(max(0, x), raw_image_width), min(max(0, y), raw_image_height)
    raw_right_down = min(max(0, x+w), raw_image_width), min(max(0, y+h), raw_image_height)
    target_left_up = raw_left_up[0] - x, raw_left_up[1] - y
    taget_right_down = raw_right_down[0] - x, raw_right_down[1] - y
    raw_image = raw_image.copy()
    raw_image[raw_left_up[1]:raw_right_down[1], raw_left_up[0]:raw_right_down[0], :] = target_image[target_left_up[1]:taget_right_down[1], target_left_up[0]:taget_right_down[0], :]
    return raw_image