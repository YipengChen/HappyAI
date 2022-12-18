import cv2
import numpy as np
from ..image import overlap_image

class HandGestureRecognitionMediaPipe(object):

    def __init__(self):
        # refer to https://github.com/YipengChen/TrainAndDeployDNNClassficationModel/tree/main/tasks/hand_gesture_recognition
        # hand_gesture_recognition_v1, input:n*42, output:n*7 (0: five; 1:ok; 2:six; 3: thumb_up; 4: thumb_down; 5:fist; 6:yeah)
        self.net = cv2.dnn.readNetFromONNX('./models/hand_gesture_recognition_v1.onnx')
        self.labels = {0:'five', 1:'ok', 2:'six', 3:'thumb_up', 4:'thumb_down', 5:'fist', 6:'yeah'}
        self.gesture_images = {
            0: cv2.imread('./images/hand_gesture_recognition/five.png'),
            1: cv2.imread('./images/hand_gesture_recognition/ok.png'),
            2: cv2.imread('./images/hand_gesture_recognition/six.png'),
            3: cv2.imread('./images/hand_gesture_recognition/thumb_up.png'),
            4: cv2.imread('./images/hand_gesture_recognition/thumb_down.png'),
            5: cv2.imread('./images/hand_gesture_recognition/fist.png'),
            6: cv2.imread('./images/hand_gesture_recognition/yeah.png'),
        }
    
    # refer to https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    def pre_process_landmark(self, landmarks):
        base_x, base_y = landmarks[0, 0], landmarks[0, 1]
        landmarks[:, 0] = landmarks[:, 0] - base_x 
        landmarks[:, 1] = landmarks[:, 1] - base_y
        flatten_landmarks = landmarks.flatten() # 42
        max_value = np.abs(flatten_landmarks).max()
        result = flatten_landmarks / max_value # normalization
        return result[None, :]

    def inference(self, landmarks):
        inputs = self.pre_process_landmark(landmarks)
        self.net.setInput(inputs)
        outputs = np.argmax(self.net.forward())
        return outputs

    def draw(self, image, gesture_type, location):
        gesture_image = self.gesture_images[gesture_type]
        return overlap_image(image, gesture_image, location)


class HandGestureRecognition(HandGestureRecognitionMediaPipe):
    pass