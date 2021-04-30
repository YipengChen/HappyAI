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
        right_eye_feature_index = [362, 385, 387, 263, 373, 380, 362]

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

    def calculation_distance(self, point1, point2):
        #return np.sqrt(np.sum(np.square(point1 - point2)))
        return np.linalg.norm(point1-point2)  