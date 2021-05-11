import cv2
import numpy as np

class StyleTransfer(object):

    style_dict = {
        'mosaic': '../models/mosaic-9.onnx',
        'candy': '../models/candy-9.onnx',
        'rain-princess': '../models/rain-princess-9.onnx',
        'udnie': '../models/udnie-9.onnx',
        'pointilism': '../models/pointilism-9.onnx'
    }

    def __init__(self, style='mosaic', method='opencv_dnn'):
        if method == 'opencv_dnn':
            onnx_file = self.style_dict[style]
            self.net = cv2.dnn.readNetFromONNX(onnx_file)
            self.inference_engine = self.net
        self.method = method
    
    def inference(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255)
        self.inference_engine.setInput(blob)
        results = self.inference_engine.forward()
        results = np.clip(results, 0, 255)
        return results.squeeze().transpose(1,2,0).astype("uint8")