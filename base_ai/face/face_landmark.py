import os
import numpy as np
import onnxruntime
import cv2


# Method 2: mediapipe, https://github.com/google/mediapipe (v0.9.0), convert tflite to onnx and use onnx runtime
class FaceLandmarkMediaPipeOnnx(object):

    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), "face_landmark.onnx"))
        self.input_height, self.input_width = 192, 192

    def _preprocess(self, image):
        input_image = image.copy()
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))
        input_image = np.array(input_image[None, :, :, :], dtype=np.float32)/255.0
        return input_image

    def _postprocess(self, score, landmarks, image_shape): 
        landmarks = landmarks.reshape(468, 3)
        landmarks[:, 0] = landmarks[:, 0] * image_shape[1] / self.input_width
        landmarks[:, 1] = landmarks[:, 1] * image_shape[0] / self.input_height
        landmarks[:, 2] = landmarks[:, 2] * 1000
        return np.int16(landmarks)

    def _inference(self, image):
        input_image = self._preprocess(image)
        score, landmarks = self.onnx_session.run(['conv2d_31', 'conv2d_21'], {'input_1':input_image})
        return self._postprocess(score, landmarks, image.shape)
        
    def inference(self, image):
        return self._inference(image)

    def draw(self, image, results):
        draw_image = image.copy()
        for landmarks in results:
            draw_image = cv2.circle(draw_image, (landmarks[0], landmarks[1]), 3, (255, 0, 0), 1)
        return draw_image


class FaceLandmark(FaceLandmarkMediaPipeOnnx):
    pass


if __name__ == '__main__':

    face_landmark = FaceLandmarkMediaPipeOnnx()
    image = cv2.imread('hand.jpg')
    draw_image = cv2.resize(image, (224, 224))
    result = hand_landmark.inference(image)
    draw_image = hand_landmark.draw(draw_image, result)
    cv2.imwrite('hand_result.jpg', draw_image)
    
