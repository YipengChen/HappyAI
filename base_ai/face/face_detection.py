import cv2
import mediapipe as mp
import numpy as np

# Method 1: mediapipe, https://github.com/google/mediapipe
class FaceDetectionMediaPipe(object):

    def __init__(self):
        self.min_detection_confidence = 0.5 # not work
        self.model_selection = 1 # 0: works best for faces within 2 meters from the camera , 1: for a full-range model best for faces within 5 meters
        self.inference_engine = mp.solutions.face_detection.FaceDetection(model_selection=self.model_selection, min_detection_confidence=self.min_detection_confidence)
        self.draw_engine = mp.solutions.drawing_utils

    def inference(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(input_image)
        return results
                
    def draw(self, image, results):
        draw_image = image.copy()
        if results.detections:
            for detection in results.detections:
                # detection.label_id, detection.score, detection.location_data
                self.draw_engine.draw_detection(draw_image, detection)
        return draw_image

# Choose a method
class FaceDetection(FaceDetectionMediaPipe):
    pass
