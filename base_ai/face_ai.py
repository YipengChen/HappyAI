import cv2
import mediapipe as mp
import numpy as np


class FaceDetection(object):

    def __init__(self, method='mediapipe'):
        if method == 'mediapipe':
            self.inference_engine = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            self.draw_engine = mp.solutions.drawing_utils
        self.method = method
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        if results.detections:
            image = image.copy()
            for detection in results.detections:
                self.draw_engine.draw_detection(image, detection)
        return image


class FaceMesh(object):

    def __init__(self, method='mediapipe'):
        if method == 'mediapipe':
            self.inference_engine = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.draw_engine = mp.solutions.drawing_utils
        self.method = method
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        if results.multi_face_landmarks:
            image = image.copy()
            drawing_spec = self.draw_engine.DrawingSpec(thickness=1, circle_radius=1)
            for face_landmarks in results.multi_face_landmarks:
                self.draw_engine.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp.solutions.face_mesh.FACE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        return image

    def get_lip_location(sell, results, image_shape):
        # Clockwise
        upper_lip_index = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 306, 292, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 62, 76]
        lower_lip_index = [61, 72, 62, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 292, 306, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            face_landmarks = np.transpose(np.stack((x, y))) * [image_shape[1], image_shape[0]]
            face_landmarks = face_landmarks.astype(np.int32)
            return face_landmarks[upper_lip_index], face_landmarks[lower_lip_index]

        return None, None

    def get_mask_location(self, results, image_shape):
        # Clockwise
        #mask_index = [93, 6, 323, 365, 152, 136]
        mask_index = [118, 197, 347, 422, 152, 202]

        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            face_landmarks = np.transpose(np.stack((x, y))) * [image_shape[1], image_shape[0]]
            return face_landmarks[mask_index]
        
        return None

    def mouth_opened_detection(self, results, image_shape, threshold=0.2):
        # Clockwise
        mouth_feature_index = [78, 82, 312, 308, 317, 87]

        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            face_landmarks = np.transpose(np.stack((y, x))) * image_shape[:2]

            # left eye
            mouth_feature_location = face_landmarks[mouth_feature_index]
            mouth_horizontal_distance = self.calculation_distance(mouth_feature_location[0], mouth_feature_location[3])
            mouth_vertical_distance = (self.calculation_distance(mouth_feature_location[1], mouth_feature_location[5]) + 
                self.calculation_distance(mouth_feature_location[2], mouth_feature_location[4])) /2
            mouth_distance_ratio = mouth_vertical_distance / mouth_horizontal_distance
        
            if mouth_distance_ratio > threshold:
                return True  

        return False      

    def eyes_closed_detection(self, results, image_shape, threshold=0.2):
        # Clockwise
        left_eye_feature_index = [33, 160, 158, 133, 153, 144]
        right_eye_feature_index = [362, 385, 387, 263, 373, 380]

        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            face_landmarks = np.transpose(np.stack((y, x))) * image_shape[:2]

            # left eye
            left_eye_feature_location = face_landmarks[left_eye_feature_index]
            left_eye_horizontal_distance = self.calculation_distance(left_eye_feature_location[0], left_eye_feature_location[3])
            left_eye_vertical_distance = (self.calculation_distance(left_eye_feature_location[1], left_eye_feature_location[5]) + 
                self.calculation_distance(left_eye_feature_location[2], left_eye_feature_location[4])) /2
            left_eye_distance_ratio = left_eye_vertical_distance / left_eye_horizontal_distance
            #print(left_eye_distance_ratio)

            # right eye
            right_eye_feature_location = face_landmarks[right_eye_feature_index]
            right_eye_horizontal_distance = self.calculation_distance(right_eye_feature_location[0], right_eye_feature_location[3])
            right_eye_vertical_distance = (self.calculation_distance(right_eye_feature_location[1], right_eye_feature_location[5]) + 
                self.calculation_distance(right_eye_feature_location[2], right_eye_feature_location[4])) /2
            right_eye_distance_ratio = right_eye_vertical_distance / right_eye_horizontal_distance
            #print(right_eye_distance_ratio)

            if left_eye_distance_ratio < threshold and right_eye_distance_ratio < threshold:
                return True
        return False

    def eyes_enlarged(self, results, image, enlarge_factor):
        # Clockwise
        left_eye_feature_index = [33, 160, 158, 133, 153, 144]
        right_eye_feature_index = [362, 385, 387, 263, 373, 380]

        new_image = image.copy()

        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            face_landmarks = np.transpose(np.stack((y, x))) * image.shape[:2]

            left_eye_feature_location = face_landmarks[left_eye_feature_index]
            right_eye_feature_location = face_landmarks[right_eye_feature_index]
            left_eye_center_location = np.mean(left_eye_feature_location, axis=0)
            right_eye_center_location = np.mean(right_eye_feature_location, axis=0)

            r_max = int(self.calculation_distance(left_eye_feature_location, right_eye_feature_location) / 4)

            '''
            # slow
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    raw_point = np.array([i, j])
                    
                    left_r = self.calculation_distance(left_eye_center_location, raw_point)
                    if left_r < r_max:
                        factor = (1.0 - np.power((left_r/r_max -1.0),2.0) * enlarge_factor)
                        new_point = (left_eye_center_location + (raw_point - left_eye_center_location) * factor).astype(np.int)
                        new_image[i, j] = image[new_point[0], new_point[1]]

                    right_r = self.calculation_distance(right_eye_center_location, raw_point)
                    if right_r < r_max:
                        factor = (1.0 - np.power((right_r/r_max -1.0),2.0) * enlarge_factor)
                        new_point = (right_eye_center_location + (raw_point - right_eye_center_location) * factor).astype(np.int)
                        new_image[i, j] = image[new_point[0], new_point[1]]
            '''

            # quick
            x_grid, y_grid = np.meshgrid(np.arange(0, image.shape[0], 1), np.arange(0, image.shape[1], 1), indexing='ij')
            xy_grid = np.transpose(np.stack((x_grid.flatten(), y_grid.flatten())))
            
            left_eye_distances = np.linalg.norm(left_eye_center_location - xy_grid, axis=1)
            left_eye_indexs = (left_eye_distances < r_max)
            enlarge_factors = 1.0 - np.power((left_eye_distances[left_eye_indexs]/r_max -1.0),2.0) * enlarge_factor
            enlarge_factors = np.transpose(np.stack((enlarge_factors,enlarge_factors)))
            left_eye_new_locations = (left_eye_center_location + (xy_grid[left_eye_indexs] - left_eye_center_location) * enlarge_factors).astype(np.int)
            left_vaild_indexs = (left_eye_new_locations[:, 0] < image.shape[0]) * (left_eye_new_locations[:, 1] < image.shape[1])
            new_image[xy_grid[left_eye_indexs][left_vaild_indexs, 0], xy_grid[left_eye_indexs][left_vaild_indexs, 1]] = image[left_eye_new_locations[left_vaild_indexs, 0], left_eye_new_locations[left_vaild_indexs, 1]]

            right_eye_distances = np.linalg.norm(right_eye_center_location - xy_grid, axis=1)
            right_eye_indexs = (right_eye_distances < r_max)
            enlarge_factors = 1.0 - np.power((right_eye_distances[right_eye_indexs]/r_max -1.0),2.0) * enlarge_factor
            enlarge_factors = np.transpose(np.stack((enlarge_factors,enlarge_factors)))
            right_eye_new_locations = (right_eye_center_location + (xy_grid[right_eye_indexs] - right_eye_center_location) * enlarge_factors).astype(np.int)
            right_vaild_indexs = (right_eye_new_locations[:, 0] < image.shape[0]) * (right_eye_new_locations[:, 1] < image.shape[1])
            new_image[xy_grid[right_eye_indexs][right_vaild_indexs, 0], xy_grid[right_eye_indexs][right_vaild_indexs, 1]] = image[right_eye_new_locations[right_vaild_indexs, 0], right_eye_new_locations[right_vaild_indexs, 1]]

        return new_image


    def calculation_distance(self, point1, point2):
        #return np.sqrt(np.sum(np.square(point1 - point2)))
        return np.linalg.norm(point1-point2)  