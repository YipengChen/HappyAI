import cv2
import mediapipe as mp
import numpy as np


class SelfieSegmentation(object):

    def __init__(self, method='mediapipe'):
        if method == 'mediapipe':
            self.inference_engine = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            self.draw_engine = None
        self.method = method
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        output_image = np.where(condition, image, bg_image)
        return output_image