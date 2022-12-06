import cv2
import mediapipe as mp


class PoseDetectionMediaPipe(object):

    def __init__(self):
        self.inference_engine = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.draw_engine = mp.solutions.drawing_utils
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        image = image.copy()
        self.draw_engine.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return image


class PoseDetection(PoseDetectionMediaPipe):
    pass