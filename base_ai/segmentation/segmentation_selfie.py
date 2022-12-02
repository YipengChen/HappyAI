import cv2
import mediapipe as mp
import numpy as np
import onnxruntime


# https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg
class SegmentationSelfiePPHunmanSegV2(object):
    def __init__(self):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession("./models/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.onnx")
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name
        self.input_height = 144
        self.input_width = 256
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,1,3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,1,3)

    def inference(self, image):
        # preprocess
        input_image = image.copy()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))
        input_image = (input_image.astype(np.float32) / 255.0 - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)
        # onnxruntime
        result = self.onnx_session.run([self.output_name], {self.input_name: input_image[None,:,:,:]})
        # postprocess
        segmentation_map = result[0][0, 1, :, :].squeeze()
        segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]))
        return segmentation_map

    def draw(self, image, results):
        image = image.copy()
        condition = np.stack((results,) * 3, axis=-1)
        output_image = np.uint8(condition * image)
        return output_image


class SegmentationSelfieMediapipe(object):

    def __init__(self):
        self.inference_engine = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.draw_engine = None
        self.method = method
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1)
        output_image = np.uint8(condition * image)
        return output_image


class SegmentationSelfie(SegmentationSelfiePPHunmanSegV2):
    pass