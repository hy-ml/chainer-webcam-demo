import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import chainer

from setup_helper import setup_model, setup_capture, setup_visualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID. GPU ID < 0 means CPU.')
    parser.add_argument('--cap', type=str, default='webcam',
                        help='Capture type.', choices=['webcam'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = setup_model(args.model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    if 'YOLO' in args.model:
        pretrained_dataset = 'voc'
    elif 'RCNN' in args.model:
        pretrained_dataset = 'coco'
    else:
        raise ValueError('Not support model `{}`.'.format(args.model))
    visualizer = setup_visualizer(args.task, pretrained_dataset)

    cap = setup_capture(args.cap)
    cap.start_device()
    for _ in range(1000):
        frame = cap.get_frame().astype(np.float32)
        frame = frame[:, :, ::-1]  # GBR -> RGB
        frame = frame.transpose((2, 0, 1))  # HWC -> CHW

        outputs = model.predict([frame])
        visualizer.visualize(frame, outputs)
        plt.show()

    plt.close()
    cap.stop_device()


if __name__ == '__main__':
    main()
