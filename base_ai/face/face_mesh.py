import cv2
import mediapipe as mp
import numpy as np

# Method_1: mediapipe, https://github.com/google/mediapipe
class FaceMeshMediaPipe(object):

    def __init__(self, max_num_faces=1):
        self.static_image_mode = False
        self.max_num_faces = max_num_faces
        self.refine_landmarks = True
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5
        self.inference_engine = mp.solutions.face_mesh.FaceMesh(
            static_image_mode = self.static_image_mode,
            max_num_faces = self.max_num_faces,
            refine_landmarks = self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)
        self.draw_engine = mp.solutions.drawing_utils
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        if results.multi_face_landmarks:
            image = image.copy()
            drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            for face_landmarks in results.multi_face_landmarks:
                self.draw_engine.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=drawing_spec)
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

    def get_ar_location(self, results, image_shape):
        ar_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                178, 162, 54, 67, 10, 297, 284, 389]

        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            face_landmarks = np.transpose(np.stack((x, y))) * [image_shape[1], image_shape[0]]
            return face_landmarks[ar_keypoint_indices]
        
        return None

    def get_all_location(self, results, image_shape, has_z=False):
        # 只选取第一个人脸
        face_landmarks = []
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            x = [landmark.x for landmark in landmarks.landmark]
            y = [landmark.y for landmark in landmarks.landmark]
            if has_z:
                z = [landmark.z for landmark in landmarks.landmark]
                face_landmarks = np.transpose(np.stack((x, y, z))) * [image_shape[1], image_shape[0], 1000]
            else:
                face_landmarks = np.transpose(np.stack((x, y))) * [image_shape[1], image_shape[0]]
            return np.around(face_landmarks,0)
        return None

    def get_all_locations(self, results, image_shape, has_z=False):
        face_landmarks = []
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                x = [landmark.x for landmark in landmarks.landmark]
                y = [landmark.y for landmark in landmarks.landmark]
                if has_z:
                    z = [landmark.z for landmark in landmarks.landmark]
                    single_face_landmarks = np.transpose(np.stack((x, y, z))) * [image_shape[1], image_shape[0], 1000]
                else:
                    single_face_landmarks = np.transpose(np.stack((x, y))) * [image_shape[1], image_shape[0]]
                single_face_landmarks = np.around(single_face_landmarks, 0)
                face_landmarks.append(single_face_landmarks)
            return face_landmarks
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
            left_eye_center_location = np.mean(left_eye_feature_location, axis=0).astype(np.int)
            right_eye_center_location = np.mean(right_eye_feature_location, axis=0).astype(np.int)

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
            for eye_center_location in [left_eye_center_location, right_eye_center_location]:
                x_grid_range = np.arange(max(eye_center_location[0]-r_max, 0), min(eye_center_location[0]+r_max+1, image.shape[0]), 1)
                y_grid_range = np.arange(max(eye_center_location[1]-r_max, 0), min(eye_center_location[1]+r_max+1, image.shape[1]), 1)
                x_grid, y_grid = np.meshgrid(x_grid_range, y_grid_range, indexing='ij')
                xy_grid = np.transpose(np.stack((x_grid.flatten(), y_grid.flatten())))
                
                eye_distances = np.linalg.norm(eye_center_location - xy_grid, axis=1)
                eye_indexs = (eye_distances < r_max)
                enlarge_factors = 1.0 - np.power((eye_distances[eye_indexs]/r_max -1.0),2.0) * enlarge_factor
                enlarge_factors = np.transpose(np.stack((enlarge_factors,enlarge_factors)))
                eye_new_locations = (eye_center_location + (xy_grid[eye_indexs] - eye_center_location) * enlarge_factors).astype(np.int)
                vaild_indexs = (eye_new_locations[:, 0] < image.shape[0]) * (eye_new_locations[:, 1] < image.shape[1])
                new_image[xy_grid[eye_indexs][vaild_indexs, 0], xy_grid[eye_indexs][vaild_indexs, 1]] = image[eye_new_locations[vaild_indexs, 0], eye_new_locations[vaild_indexs, 1]]

        return new_image


    def calculation_distance(self, point1, point2):
        #return np.sqrt(np.sum(np.square(point1 - point2)))
        return np.linalg.norm(point1-point2)


# Choose a method
class FaceMesh(FaceMeshMediaPipe):
    pass