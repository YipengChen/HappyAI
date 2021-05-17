import cv2
import numpy as np

class StyleTransfer(object):

    style_dict = {
        'mosaic': '../models/style_transfer/mosaic-9.onnx',
        'candy': '../models/style_transfer/candy-9.onnx',
        'rain-princess': '../models/style_transfer/rain-princess-9.onnx',
        'udnie': '../models/style_transfer/udnie-9.onnx',
        'pointilism': '../models/style_transfer/pointilism-9.onnx',
        'cartoon': '../models/style_transfer/cartoon.pb'
    }

    def __init__(self, style='mosaic', method='opencv_dnn'):
        if method == 'opencv_dnn':
            if style == 'cartoon':
                pb_file = self.style_dict[style]
                self.net = cv2.dnn.readNetFromTensorflow(pb_file)
                self.inference_engine = self.net
            else:
                onnx_file = self.style_dict[style]
                self.net = cv2.dnn.readNetFromONNX(onnx_file)
                self.inference_engine = self.net
        self.style = style
        self.method = method
    
    def inference(self, image):
        if self.style == 'cartoon':
            input_image = image.copy().astype(np.float32) / 127.5 - 1
            blob = cv2.dnn.blobFromImage(input_image)
            self.inference_engine.setInput(blob)
            results = self.inference_engine.forward()
            results = (results + 1) * 127.5
            results = np.clip(results, 0, 255)
            results = results.squeeze().transpose(1,2,0).astype("uint8")
            #return cv2.ximgproc.guidedFilter(image,results,1,100)
            return results
        else:
            blob = cv2.dnn.blobFromImage(image, 1/255)
            self.inference_engine.setInput(blob)
            results = self.inference_engine.forward()
            results = np.clip(results, 0, 255)
            return results.squeeze().transpose(1,2,0).astype("uint8")