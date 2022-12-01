import cv2
import numpy as np

class ObjectDetection_20class(object):

    def __init__(self, method='opencv_dnn'):
        if method == 'opencv_dnn':
            prototxt_file = '../models/MobileNetSSD_deploy.prototxt'
            model_file = '../models/MobileNetSSD_deploy.caffemodel'
            self.net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
            self.inference_engine = self.net
        self.method = method
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                    "sofa", "train", "tvmonitor"]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

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
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), self.colors[idx], 2)
                cv2.putText(image, label, (start_x, start_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[idx], 2)
        return image


class ObjectDetection_80class(object):

    def __init__(self, method='opencv_dnn', model_name='yolov4-tiny', confidence_threshold=0.25, nms_threshold=0.4):
        if method == 'opencv_dnn':
            if model_name == 'yolov4-tiny':
                cfg_file = '../models/yolov4-tiny.cfg'
                weights_file = '../models/yolov4-tiny.weights'
                net = cv2.dnn.readNet(weights_file, cfg_file)
                model = cv2.dnn_DetectionModel(net)
                model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
            elif model_name == 'yolo-fastest':
                cfg_file = '../models/yolo-fastest-1.1.cfg'
                weights_file = '../models/yolo-fastest-1.1.weights'
                net = cv2.dnn.readNet(weights_file, cfg_file)
                model = cv2.dnn_DetectionModel(net)
                model.setInputParams(size=(320, 320), scale=1/255, swapRB=True)
            self.inference_engine = model
        self.method = method
        with open("../models/coco.names", "r") as f:
            self.classes = [cname.strip() for cname in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def inference(self, image):
        classes, scores, boxes = self.inference_engine.detect(image, self.confidence_threshold, self.nms_threshold)
        return [classes, scores, boxes]

    def draw(self, image, results, confidence_threshold=0.25):
        classes, scores, boxes = results
        for class_index, score, box in zip(classes, scores, boxes):
            if score > confidence_threshold:
                start_x, start_y, width, height = box
                label = "{}: {:.2f}%".format(self.classes[class_index], score * 100)
                cv2.rectangle(image, box, self.colors[class_index], 2)
                cv2.putText(image, label, (start_x, start_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_index], 2)
        return image