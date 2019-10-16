class VisualizerBase(object):
    """Base class of all visualizer classes.

    """

    def visualize(self, frame, outputs):
        raise NotImplementedError()
