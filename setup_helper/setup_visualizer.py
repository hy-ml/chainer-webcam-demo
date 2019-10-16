from visualizer import DetectionVisualizer


def setup_visualizer(task, pretrained_dataset):
    if task == 'detection':
        assert pretrained_dataset in ['voc', 'coco']
        visualizer = DetectionVisualizer(pretrained_dataset)
    else:
        raise ValueError('Not support visualizer of `{}`.'.format(task))
    return visualizer
