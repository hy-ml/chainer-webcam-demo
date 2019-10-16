class CaptureBase(object):
    """Base class of all `Capture` classes.

    """

    def start_device(self):
        raise NotImplementedError()

    def stop_device(self):
        raise NotADirectoryError()

    def get_frame(self):
        raise NotImplementedError()
