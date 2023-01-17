import cv2
import mediapipe as mp
import numpy as np
import onnxruntime


# https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg

# how to convert PP model to onnx
# 1. 使用工具固定输入shape（可选）（https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tools/paddle/README.md）
# 1.1 python paddle_infer_shape.py --model_dir XXXX//portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax  \
#                              --model_filename model.pdmodel \
#                              --params_filename model.pdiparams \
#                              --save_dir new_model \
#                              --input_shape_dict="{'x':[1,3,144,256]}"
# 2. 使用paddle2onnx将paddle模型转为onnx (https://github.com/PaddlePaddle/PaddleSeg/blob/19351bab9a824a8f96e1c1b527ec2d7db21309c9/docs/model_export_onnx_cn.md)
# 2.1. pip install paddlepaddle, paddleseg, paddle2onnx
# 2.2. paddle2onnx --model_dir output \
#             --model_filename model.pdmodel \
#             --params_filename model.pdiparams \
#             --opset_version 11 \
#             --save_file output.onnx

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


# https://github.com/PeterL1n/RobustVideoMatting
class SegmentationSelfieRobustVideoMatting(object):

    def __init__(self, threshold=0.5):
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession("./models/rvm_mobilenetv3_fp32.onnx")
        self.downsample_ratio = np.array([0.8], dtype=np.float32)  # dtype always FP32
        self.rec = [ np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # Must match dtype of the model.
        self.input_height = 480
        self.input_width = 640
        self.threshold = threshold

    def reset(self):
        self.rec = [ np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # Must match dtype of the model.

    def inference(self, image):
        # preprocess
        input_image = image.copy()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.array(input_image[None, :, :, :], dtype=np.float32)/255.0
        # onnxruntime
        fgr, pha, *rec = self.onnx_session.run([], {
            'src': input_image,
            'r1i': self.rec[0],
            'r2i': self.rec[1],
            'r3i': self.rec[2],
            'r4i': self.rec[3],
            'downsample_ratio': self.downsample_ratio
        })
        self.rec = rec
        # postprocess
        segmentation_map = pha.squeeze()
        segmentation_map = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]))
        segmentation_map[segmentation_map>=self.threshold] = 1
        segmentation_map[segmentation_map<self.threshold] = 0
        segmentation_map = segmentation_map.astype(np.uint8)
        return segmentation_map

    def draw(self, image, results):
        image = image.copy()
        condition = np.stack((results,) * 3, axis=-1)
        output_image = np.uint8(condition * image)
        return output_image



class SegmentationSelfie(SegmentationSelfieRobustVideoMatting):
    pass