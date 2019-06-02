import torch
import torch.nn as nn
from src.anchor_generation import generate_anchor_base
from src.proposal_layer import ProposalLayer  
from torch.nn import functional as F

class RegionProposalLayers(nn.Module):
    '''
    region proposal network in faster rcnn
    '''
    def __init__(self, in_channel=512, mid_channel=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16,
                 proposal_creator_params=dict()
    ):

        super(RegionProposalLayers, self).__init__()
        self.feat_stride = feat_stride
        # define the conv layer to processing input feature map
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1, bias=True)
        # generate basic anchors
        self.anchor_base = generate_anchor_base.chenyun_method(ratios=ratios, anchor_scales=anchor_scales)
        self.n_anchor = self.anchors_base.shape[0]
        # define bg/fg classificatinon score layer
        self.score = nn.Conv2d(mid_channel, self.n_anchor * 2, 1, 1, 0) # 2(bg/fg) * 9 (anchors)
        # define anchor box offset prediction layer
        self.loc = nn.Conv2d(mid_channel, self.n_anchor * 4, 1, 1, 0) # 4(coords) * 9 (anchors)
        # define proposal layer
        self.proposal_layer = ProposalLayer(self, **proposal_creator_params)
        # initialize parameters
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            -1,
            input_shape[3]
        )
        return x

    def forward(self, base_feature, img_size, min_scale=1.):
        '''
        Arg:
            base_feature: (N,C,H,W)
            min_scale: default minimum size=16, you can ues scale to adjust min_size: min_size*scale;
        '''
        # input the first conv(3X3)
        begin_feature = F.relu(self.conv1(base_feature))

        # forward the bg/fg classifier layer
        rpn_scores = self.score(begin_feature)
        rpn_scores_reshape = self.reshape(rpn_scores, 2)    # from (n,2*9,h,w) to (n,2,9*h,w)
        rpn_scores_prob_reshape = F.softmax(rpn_scores_reshape, 1)
        rpn_scores_prob = self.reshape(rpn_scores_prob_reshape, rpn_scores.size()[1])   # from (n,2,9*h,w) to (n,2*9,h,w)

        # forward the bounding box regression layer: get offsets to the anchor boxes: (n,4*9,h,w)
        rpn_locs_pred = self.loc(begin_feature)

        # forward proposal layer to get rois and anchors shifted on the image
        batch_size, _, hh, ww = base_feature.shape
        feature_shape = (hh, ww)
        rois, anchor = self.proposal_layer(rpn_scores_prob.data, rpn_locs_pred.data, self.anchor_base, 
                                           batch_size, feature_shape, img_size, min_scale)
        
        return rpn_scores_prob, rpn_locs_pred, rois, anchor 

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()