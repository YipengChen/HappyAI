import os
import numpy as np
import onnxruntime
import cv2


# Method 2: mediapipe, https://github.com/google/mediapipe (v0.9.0), convert tflite to onnx and use onnx runtime
class HandLandmarkMediaPipeOnnx(object):

    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), "hand_landmark_lite.onnx"))
        self.input_height, self.input_width = 224, 224

    def _preprocess(self, image):
        input_image = image.copy()
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))
        input_image = np.array(input_image[None, :, :, :], dtype=np.float32)/255.0
        return input_image

    def _postprocess(self, left_score, left_landmarks, right_score, right_landmarks, image_shape): 
        right_landmarks = right_landmarks.reshape(21, 3)
        right_landmarks[:, 0] = right_landmarks[:, 0] * image_shape[1] / self.input_width
        right_landmarks[:, 1] = right_landmarks[:, 1] * image_shape[0] / self.input_height
        return np.int16(right_landmarks)

    def _inference(self, image):
        input_image = self._preprocess(image)
        left_score, left_landmarks, right_score, right_landmarks = self.onnx_session.run(['Identity_1', 'Identity_3', 'Identity_2', 'Identity'], {'input_1':input_image})
        print(left_score, left_landmarks, right_score, right_landmarks)
        return self._postprocess(left_score, left_landmarks, right_score, right_landmarks, image.shape)
        
    def inference(self, image):
        return self._inference(image)

    def draw(self, image, results):
        draw_image = image.copy()
        for landmarks in results:
            draw_image = cv2.circle(draw_image, (landmarks[0], landmarks[1]), 3, (255, 0, 0), 1)
        return draw_image


class HandLandmark(HandLandmarkMediaPipeOnnx):
    pass


if __name__ == '__main__':

    hand_landmark = HandLandmarkMediaPipeOnnx()
    image = cv2.imread('hand.jpg')
    draw_image = cv2.resize(image, (224, 224))
    result = hand_landmark.inference(image)
    draw_image = hand_landmark.draw(draw_image, result)
    cv2.imwrite('hand_result.jpg', draw_image)
    
