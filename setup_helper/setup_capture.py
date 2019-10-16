from capture import WebCam


def setup_capture(cap_type):
    if cap_type == 'webcam':
        cap = WebCam()
    else:
        raise ValueError('Not support `cap_type`: `{}`.'.format(cap_type))
    return cap
