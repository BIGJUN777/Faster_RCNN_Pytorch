# code based on chenyun
# https://github.com/chenyuntc/simple-faster-rcnn-pytorch

import numpy as np
import six

class generate_anchor_base:
    '''
    There are two methods to generate the basic anchors. The results are almost similiar.
    One function is written by chenyun named chenyun_method();
    The other one is written by Ross Girshick and Sean Bell named ross_method(); 
    '''
    def __init__(self):
        pass
    @staticmethod
    def chenyun_method( base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        """
        Generate anchor base windows by enumerating aspect ratio and scales.
        
        EQUATION:   
                    enumerate aspect ratio: w'= np.sqrt(base_size**2 / ratios)
                                            w'= base_size * np.sqrt(1 / ratios)
                    enumerate aspect scales: w = w' * anchor_scales
                    height:width = {0.5, 1, 2} --> h = w * ratio

        Args:
            base_size (number): The width and the height of the reference window.
            ratios (list of floats): This is ratios of width to height of
                the anchors.
            anchor_scales (list of numbers): This is areas of anchors.
                Those areas will be the product of the square of an element in
                :obj:`anchor_scales` and the original area of the reference
                window.

        Returns:
            ~numpy.ndarray:
            An array of shape :math:`(R, 4)`.
            Each element is a set of coordinates of a bounding box.
            The second axis corresponds to
            :math:`(x_{min}, y_{min}, x_{max}, y_{max})` of a bounding box.

        """
        py = base_size / 2.
        px = base_size / 2.

        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                            dtype=np.float32)
        for i in six.moves.range(len(ratios)):
            for j in six.moves.range(len(anchor_scales)):
                w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
                h = base_size * anchor_scales[j] * np.sqrt(ratios[i])

                index = i * len(anchor_scales) + j
                anchor_base[index, 1] = np.round(py - h / 2.)
                anchor_base[index, 0] = np.round(px - w / 2.)
                anchor_base[index, 3] = np.round(py + h / 2.)
                anchor_base[index, 2] = np.round(px + w / 2.)
        return anchor_base

###############################################################################
# Written by Ross Girshick and Sean Bell
###############################################################################
    @staticmethod
    def ross_method(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6)):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales wrt a reference (0, 0, 15, 15) window.
        """
        try:
            xrange          # Python 2
        except NameError:
            xrange = range  # Python 3

        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = _ratio_enum(base_anchor, ratios)
        anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                             for i in xrange(ratio_anchors.shape[0])])
        return anchors

# The following methods will be use by ross_method() function
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors