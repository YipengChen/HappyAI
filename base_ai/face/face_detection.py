import os
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime
from .detect_utils import overlap_similarity

# Method 1: mediapipe, https://github.com/google/mediapipe
class FaceDetectionMediaPipe(object):

    def __init__(self):
        self.min_detection_confidence = 0.75 # not work
        self.model_selection = 1 # 0: works best for faces within 2 meters from the camera , 1: for a full-range model best for faces within 5 meters
        self.inference_engine = mp.solutions.face_detection.FaceDetection(model_selection=self.model_selection, min_detection_confidence=self.min_detection_confidence)
        self.draw_engine = mp.solutions.drawing_utils

    def inference(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(input_image)
        return results
                
    def draw(self, image, results):
        draw_image = image.copy()
        if results.detections:
            for detection in results.detections:
                # detection.label_id, detection.score, detection.location_data
                self.draw_engine.draw_detection(draw_image, detection)
        return draw_image


# Method 2: mediapipe, https://github.com/google/mediapipe, convert tflite to onnx and use onnx runtime
class FaceDetectionMediaPipeOnnx(object):

    def __init__(self):

        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), "face_detection_short_range.onnx"))
        self.anchors = np.load(os.path.join(os.path.dirname(__file__), 'anchors.npy'))
        print(self.anchors)
        self.input_height = 128
        self.input_width = 128
        self.num_anchors = 896
        self.x_scale = 128.0
        self.y_scale = 128.0
        self.h_scale = 128.0
        self.w_scale = 128.0
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.5
        self.min_suppression_threshold = 0.3

    def preprocess(self, image):
        input_image = image.copy()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))
        input_image = np.array(input_image[None, :, :, :], dtype=np.float32)/127.5 - 1
        return input_image

    def inference(self, image):
        image = self.preprocess(image)
        regressors, classificators = self.onnx_session.run(['regressors', 'classificators'], {'input':image})
        # Postprocess the raw predictions:
        detections = self._tensors_to_detections(regressors, classificators, self.anchors)
        # Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            filtered_detections.append(faces)
        return filtered_detections

    def draw(self, image, results):
        draw_image = image.copy()
        for filtered_detection in results:
            for face in filtered_detection:
                xmin = int(face[0] * image.shape[1])
                ymin = int(face[1] * image.shape[0])
                xmax = int(face[2] * image.shape[1])
                ymax = int(face[3] * image.shape[0])
                #print(xmin, ymin, xmax, ymax)
                p1_x = int(face[4] * image.shape[1])
                p1_y = int(face[5] * image.shape[0])
                p2_x = int(face[6] * image.shape[1])
                p2_y = int(face[7] * image.shape[0])
                p3_x = int(face[8] * image.shape[1])
                p3_y = int(face[9] * image.shape[0])
                p4_x = int(face[10] * image.shape[1])
                p4_y = int(face[11] * image.shape[0])
                p5_x = int(face[12] * image.shape[1])
                p5_y = int(face[13] * image.shape[0])
                p6_x = int(face[14] * image.shape[1])
                p6_y = int(face[15] * image.shape[0])
                draw_image = cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                draw_image = cv2.circle(draw_image, (p1_x, p1_y), 3, (255, 0, 0), 1)
                draw_image = cv2.circle(draw_image, (p2_x, p2_y), 3, (255, 0, 0), 1)
                draw_image = cv2.circle(draw_image, (p3_x, p3_y), 3, (255, 0, 0), 1)
                draw_image = cv2.circle(draw_image, (p4_x, p4_y), 3, (255, 0, 0), 1)
                draw_image = cv2.circle(draw_image, (p5_x, p5_y), 3, (255, 0, 0), 1)
                draw_image = cv2.circle(draw_image, (p6_x, p6_y), 3, (255, 0, 0), 1)
        return draw_image

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = np.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = x_center - w / 2.  # ymin
        boxes[..., 1] = y_center - h / 2.  # xmin 
        boxes[..., 2] = x_center + w / 2.  # ymax
        boxes[..., 3] = y_center + h / 2.  # xmax

        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        
        thresh = self.score_clipping_thresh
        raw_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
        detection_scores = 1 / (1 + np.exp(-raw_score_tensor))
        detection_scores = detection_scores.squeeze(axis=-1)
        
        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]]
            output_detections.append(np.hstack((boxes, scores[:,None])))  # n*(16+1)
        return output_detections

    def _weighted_non_max_suppression(self, detections):
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = np.argsort(detections[:, 16])
        remaining = remaining[::-1]

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)
            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.copy()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(axis=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    


# Choose a method
class FaceDetection(FaceDetectionMediaPipe):
    pass



if __name__=='__main__':
    
    face_detector = FaceDetectionMediaPipeOnnx()
    image = cv2.imread('face.jpg')
    result = face_detector.inference(image)
    image = face_detector.draw(image, result)
    cv2.imwrite('face_result.jpg', image)