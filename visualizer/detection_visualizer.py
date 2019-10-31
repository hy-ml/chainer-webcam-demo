import numpy as np
import cv2

from visualizer.visualizer_base import VisualizerBase
from chainercv.datasets import coco_bbox_label_names, voc_bbox_label_names


class DetectionVisualizer(VisualizerBase):
    """Visualizer for object detection

    """

    def __init__(self, pretrained_dataset, thickness=2):
        if pretrained_dataset == 'coco':
            self._label_names = coco_bbox_label_names
        elif pretrained_dataset == 'voc':
            self._label_names = voc_bbox_label_names
        else:
            raise ValueError(
                'Not support visualization for dataset `{}`'.format(
                    pretrained_dataset))

        self._thickness = thickness

    def visualize(self, frame, outputs):
        frame = frame.copy()
        bboxes, labels, scores = outputs
        bboxes = bboxes[0].astype(np.int)
        labels = labels[0]
        scores = scores[0]

        for bbox, label, score in zip(bboxes, labels, scores):
            self._add_bbox(frame, bbox)
            self._add_text(frame, bbox, label, score)
        return frame

    def _add_bbox(self, frame, bbox):
        pt1 = (bbox[1], bbox[0])
        pt2 = (bbox[3], bbox[2])
        cv2.rectangle(frame, pt1, pt2, (50, 50, 250), self._thickness)

    def _add_text(self, frame, bbox, label, score):
        label_name = self._label_names[label]
        font = cv2.FONT_HERSHEY_SIMPLEX

        cat_size = cv2.getTextSize(label_name, font, 0.5, 2)[0]
        cv2.rectangle(frame, (bbox[1], bbox[0] - cat_size[1] - 2),
                      (bbox[1] + cat_size[0], bbox[0] - 2), (50, 50, 250), -1)
        cv2.putText(frame, label_name, (bbox[1], bbox[0] - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
