from visualizer.visualizer_base import VisualizerBase
from chainercv.datasets import coco_bbox_label_names, voc_bbox_label_names
from chainercv.visualizations import vis_bbox


class DetectionVisualizer(VisualizerBase):
    """Visualizer for object detection

    """

    def __init__(self, pretrained_dataset):
        if pretrained_dataset == 'coco':
            self._label_names = coco_bbox_label_names
        elif pretrained_dataset == 'voc':
            self._label_names = voc_bbox_label_names
        else:
            raise ValueError(
                'Not support visualization for dataset `{}`'.format(
                    pretrained_dataset))

    def visualize(self, frame, outputs):
        bboxes, labels, scores = outputs
        bbox = bboxes[0]
        label = labels[0]
        score = scores[0]
        vis_bbox(frame, bbox, label, score, label_names=self._label_names)
