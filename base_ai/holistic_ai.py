import cv2
import mediapipe as mp


class HolisticDetection(object):

    def __init__(self, method='mediapipe'):
        if method == 'mediapipe':
            self.inference_engine = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.draw_engine = mp.solutions.drawing_utils
        self.method = method
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        image = image.copy()
        self.draw_engine.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)
        self.draw_engine.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        self.draw_engine.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        self.draw_engine.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        return image