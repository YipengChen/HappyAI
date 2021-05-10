import cv2
import numpy as np

class ObjectDetection_20class(object):

    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]
    draw_colors = np.random.uniform(0, 255, size=(len(classes), 3))

    def __init__(self, method='opencv_dnn'):
        if method == 'opencv_dnn':
            prototxt_file = '../models/MobileNetSSD_deploy.prototxt'
            model_file = '../models/MobileNetSSD_deploy.caffemodel'
            self.net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
            self.inference_engine = self.net
        self.method = method
    
    def inference(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/127.5, (300, 300), 127.5)
        self.net.setInput(blob)
        results = self.net.forward()
        return results

    def draw(self, image, results, confidence_threshold=0.8):
        h, w = image.shape[:2]
        for i in np.arange(0, results.shape[2]):
            confidence = results[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(results[0, 0, i, 1])
                start_x, start_y, end_x, end_y = (results[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(np.int)
                label = "{}: {:.2f}%".format(self.classes[idx], confidence * 100)
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), self.draw_colors[idx], 2)
                cv2.putText(image, label, (start_x, start_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.draw_colors[idx], 2)
        return image