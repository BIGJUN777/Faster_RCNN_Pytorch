# import sys
# sys.path.append("../../utilse")
import torch
import numpy as np
import cupy as cp
from utils.bbox_transform import loc2bbox, clip_boxes
from utils.nms.non_maximum_suppression import non_maximum_suppression
import ipdb

class ProposalLayer(object):
    '''
    This class is used for region proposal network.
    proposal regions are generated by calling this object.
    '''
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_box_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_box_size

    def __call__(self, locs, scores, anchor_base, batch_size, feature_shape, image_size, min_scale=1.):
        '''
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes **centered** on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        '''
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.training = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        
        # the first set of _num_anchors channels are bg probs, the second set are the fg probs
        # !NOTE:WHY
        scores = scores[:, self.parent_model.n_anchor:, :, :]

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors(batch_size, K*9, 4); the same process to rpn_score
        bbox_deltas = locs.permute(0,2,3,1).contiguous().reshape(batch_size, -1, 4)
        scores = scores.permute(0,2,3,1).contiguous().reshape(batch_size, -1)
        
        # ipdb.set_trace()
        ## 1.1 generate A anchor boxes **centered** on cell i,feature size: （batch, 9*feat_h*feat_w,4）type(torch)
        anchor = _enumerate_shifted_anchor(batch_size, np.array(anchor_base), 
                                           self.parent_model.feat_stride, feature_shape)
        ## 1.2 Convert anchors into proposal with bbox transformations.
        roi = loc2bbox(anchor, bbox_deltas)
        ## 2 Clip predicted boxes to image:just clip, the number of roi is not changed 
        roi = clip_boxes(roi, image_size, batch_size)
        ## 3 remove predicted boxes with either height or width < threshold
        min_size = self.min_size * min_scale
        ws = roi[:,:,2] - roi[:,:,0]
        hs = roi[:,:,3] - roi[:,:,1]
        # !NOTE should change to numpy???
        keep = np.where((ws.numpy() >= min_size) & (hs.numpy() >= min_size))[1]
        roi_keep = roi[:,keep,:]
        scores_keep = scores[:,keep]
        ## 4 sort all (proposal, score) pairs by score from highest to lowest
        _, order = torch.sort(scores_keep, 1, True)

        for i in range(batch_size):
            
            roi_single = roi_keep[i]
            score_single = scores_keep[i]
            order_single = order[i]

            ## 5 Take top pre_nms_topN (e.g. 6000).
            if n_pre_nms > 0 and n_pre_nms < scores_keep.numel():
                order_single = order_single[:n_pre_nms]
            roi_single = roi_single[order_single,:]
            score_single = score_single[order_single]

            # 6. apply nms (e.g. threshold = 0.7)
            keep = non_maximum_suppression(
                cp.ascontiguousarray(cp.asarray(roi_single)),
                thresh=self.nms_thresh)

            # ipdb.set_trace()
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            if n_post_nms > 0:
                keep = keep[:n_post_nms]
            roi_single = roi_single[keep,:]    

            # store roi_single
            output = roi_single 

        return output, anchor     

def _enumerate_shifted_anchor(batch_size, anchor_base, feat_stride, feature_shape):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (1, K*A, 4) shifted anchors
    # return (batch_size, K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, feature_shape[0] * feat_stride, feat_stride)
    shift_x = xp.arange(0, feature_shape[1] * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_x.ravel(), shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    # !TODO: change the center of basic anchors to see if it will affect the result or not
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    # anchor = anchor.reshape((1, K * A, 4)).astype(np.float32).expand((batch_size, -1, 4))
    anchor = torch.from_numpy(anchor)
    anchor = anchor.reshape((1, K * A, 4)).expand((batch_size, -1, 4)).float()
    return anchor
        