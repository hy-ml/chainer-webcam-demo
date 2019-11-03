import argparse
import numpy as np
import cv2
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
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()
    return args


def main():
    global flag_quit
    args = parse_args()

    model = setup_model(args.model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    if 'YOLO' in args.model or 'SSD' in args.model:
        pretrained_dataset = 'voc'
    elif 'RCNN' in args.model:
        pretrained_dataset = 'coco'
    else:
        raise ValueError('Not support model `{}`.'.format(args.model))
    visualizer = setup_visualizer(args.task, pretrained_dataset)

    cap = setup_capture(args.cap)
    cap.start_device()
    while True:
        frame = cap.get_frame()
        indata = frame.astype(np.float32)[:, :, ::-1]  # GBR -> RGB
        if args.flip:
            frame = cv2.flip(frame, 1)
            indata = cv2.flip(indata, 1)
        indata = indata.transpose((2, 0, 1))  # HWC -> CHW

        outputs = model.predict([indata])
        result = visualizer.visualize(frame, outputs)
        cv2.imshow('frame', frame)
        cv2.imshow('result', result)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.stop_device()


if __name__ == '__main__':
    main()
