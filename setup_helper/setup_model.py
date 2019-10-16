from chainercv import links


def setup_model(model_type):
    if model_type == 'YOLOv2':
        model = links.model.yolo.YOLOv2(pretrained_model='voc0712')
    elif model_type == 'YOLOv3':
        model = links.model.yolo.YOLOv3(pretrained_model='voc0712')
    elif model_type == 'FasterRCNNFPNResNet50':
        model = links.model.fpn.FasterRCNNFPNResNet50(
            pretrained_model='coco',
            return_values=['bboxes', 'labels', 'scores'])
    elif model_type == 'FasterRCNNFPNResNet101':
        model = links.model.fpn.FasterRCNNFPNResNet101(
            pretrained_model='coco',
            return_values=['bboxes', 'labels', 'scores'])
    elif model_type == 'MaskRCNNFPNResNet50':
        model = links.model.fpn.MaskRCNNFPNResNet50(
            pretrained_model='coco',
            return_values=['masks', 'labels', 'scores'])
    elif model_type == 'MaskRCNNFPNResNet101':
        model = links.model.fpn.MaskRCNNFPNResNet101(
            pretrained_model='coco',
            return_values=['masks', 'labels', 'scores'])
    else:
        raise ValueError()

    return model
