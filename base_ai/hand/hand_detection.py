import cv2
import mediapipe as mp
import numpy as np


# Method 1: mediapipe, https://github.com/google/mediapipe
class HandDetectionMediaPipe(object):

    def __init__(self):
        self.inference_engine = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.draw_engine = mp.solutions.drawing_utils
    
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

    def get_all_landmarks(self, results, image_shape, has_z=False):
        multi_hand_landmarks = []
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                x = [landmark.x for landmark in landmarks.landmark]
                y = [landmark.y for landmark in landmarks.landmark]
                if has_z:
                    z = [landmark.z for landmark in landmarks.landmark]
                    hand_landmarks = np.transpose(np.stack((x, y, z))) * [image_shape[1], image_shape[0], 1000]
                else:
                    hand_landmarks = np.transpose(np.stack((x, y))) * [image_shape[1], image_shape[0]]
                multi_hand_landmarks.append(hand_landmarks)
        return multi_hand_landmarks

class HandDetection(HandDetectionMediaPipe):
    pass