import cv2

import _init_path
from capture import WebCam


def test_webcam():
    cap = WebCam()
    cap.start_device()
    while True:
        frame = cap.get_frame()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.stop_device()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_webcam()
