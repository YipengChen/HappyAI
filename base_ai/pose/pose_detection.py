import os
import cv2
import mediapipe as mp
import onnxruntime
import numpy as np
from ..detect_utils import overlap_similarity

class PoseDetectionMediaPipe(object):

    def __init__(self):
        self.inference_engine = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.draw_engine = mp.solutions.drawing_utils
    
    def inference(self, image):
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        results = self.inference_engine.process(image)
        return results

    def draw(self, image, results):
        image = image.copy()
        self.draw_engine.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return image


# Method 2: mediapipe, https://github.com/google/mediapipe (v0.8.3.2), convert tflite to onnx and use onnx runtime
class PoseDetectionMediaPipeOnnx(object):

    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), "pose_detection.onnx"))
        self.anchors = np.load(os.path.join(os.path.dirname(__file__), 'anchors.npy'))
        self.input_height, self.input_width = 128, 128
        self.num_anchors = 896
        self.x_scale, self.y_scale, self.h_scale, self.w_scale = 128.0, 128.0, 128.0, 128.0
        self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.5
        self.min_suppression_threshold = 0.3

    def _preprocess(self, image):
        input_image = image.copy()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))
        input_image = np.array(input_image[None, :, :, :], dtype=np.float32)/127.5 - 1
        return input_image

    def _postprocess(self, regressors, classificators, image_shape):
        detections = self._tensors_to_detections(regressors, classificators, self.anchors)
        faces = self._weighted_non_max_suppression(detections[0])
        faces = [self._decode_face(face, image_shape) for face in faces]
        return faces
    
    def inference(self, image):
        input_image = self._preprocess(image)
        regressors, classificators = self.onnx_session.run(['regressors', 'classificators'], {'input':input_image})
        return self._postprocess(regressors, classificators, image.shape)
        
    def draw(self, image, results):
        draw_image = image.copy()
        for face in results:
            #draw_image = cv2.rectangle(draw_image, (face[0], face[1]), (face[2], face[3]), (255, 0, 0))
            #draw_image = cv2.circle(draw_image, (face[4], face[5]), 3, (255, 0, 0), 1)
            #draw_image = cv2.circle(draw_image, (face[6], face[7]), 3, (255, 0, 0), 1)
            height = face[7] - face[5]
            draw_image = cv2.rectangle(draw_image, (face[0], face[5] - height), (face[2], face[5] + height), (255, 0, 0))
            #draw_image = cv2.circle(draw_image, (face[8], face[9]), 3, (255, 0, 0), 1)
            #draw_image = cv2.circle(draw_image, (face[10], face[11]), 3, (255, 0, 0), 1)
        return draw_image

    def _decode_face(self, face, image_shape):
        face = np.array(face[:12])
        face[::2] = face[::2] * image_shape[1]
        face[1::2] = face[1::2] * image_shape[0]
        return np.int32(face)

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

        for k in range(4):
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
        if len(detections) == 0: return np.array([])

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = np.argsort(detections[:, 12])
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
                coordinates = detections[overlapping, :12]
                scores = detections[overlapping, 12:13]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(axis=0) / total_score
                weighted_detection[:12] = weighted
                weighted_detection[12] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    


class PoseDetection(PoseDetectionMediaPipeOnnx):
    pass