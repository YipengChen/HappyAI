import cv2
import mediapipe as mp


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