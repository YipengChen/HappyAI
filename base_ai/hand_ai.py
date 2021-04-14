import cv2
import mediapipe as mp


class HandDetection(object):

    def __init__(self, method='mediapipe'):
        if method == 'mediapipe':
            self.inference_engine = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
            self.draw_engine = mp.solutions.drawing_utils
        self.method = method
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        if results.multi_hand_landmarks:
            image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                self.draw_engine.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return image