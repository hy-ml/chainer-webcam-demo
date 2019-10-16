import cv2

from capture.capture_base import CaptureBase


class WebCam(CaptureBase):
    def __init__(self):
        self._cap = None

    def start_device(self):
        self._cap = cv2.VideoCapture(0)

    def stop_device(self):
        self._cap.release()
        del self._cap
        self._cap = None

    def get_frame(self):
        _, frame = self._cap.read()
        return frame
