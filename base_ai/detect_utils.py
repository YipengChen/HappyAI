import numpy as np

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
    box_a: (tensor) bounding boxes, Shape: [A,4].
    box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
    (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(np.repeat(box_a[:, None, 2:], B, axis=1),
                    np.repeat(box_b[None, :, 2:], A, axis=0))
    min_xy = np.maximum(np.repeat(box_a[:, None, :2], B, axis=1),
                    np.repeat(box_b[None, :, :2], A, axis=0))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))
    area_a = np.repeat(np.repeat(area_a[:, None], inter.shape[0], axis=0), inter.shape[1], axis=1)
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1]))
    area_b = np.repeat(area_b[None, :], inter.shape[0], axis=0)
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box[None,:], other_boxes).squeeze(0)