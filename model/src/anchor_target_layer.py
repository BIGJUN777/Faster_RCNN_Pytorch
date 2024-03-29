import numpy as np
from utils.bbox_transform import bbox_iou, bbox2loc
import ipdb

class AnchorTargetLayer(object):
    '''
    Assign the ground truth bounding boxes to anchors
    '''
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """
        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(batch_size, R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(batch_size, S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.
        """
        # ipdb.set_trace()
        img_H, img_W = img_size
        n_anchor = anchor.shape[1]
        # Calc indicies of shifted anchors which are located completely inside of the image
        # !NOTE, Does it necessary?
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[:,inside_index,:]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor[0], bbox[0,argmax_ious,:])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)
        
        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)
        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0
        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1
        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1
        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes : (N, K)
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1) # compare volumn by volumn
        max_ious = ious[np.arange(len(inside_index)), argmax_ious] # (N,)
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious

def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, :, 0] >= 0) &
        (anchor[:, :, 1] >= 0) &
        (anchor[:, :, 2] <= W) &
        (anchor[:, :, 3] <= H)
    )[1]
    return index_inside

