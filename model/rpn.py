import torch
import torch.nn as nn
from torch.nn import functional as F
from model.basestone import basestone
from utils.anchor_generation import generate_anchor_base
from .src.proposal_layer import ProposalLayer  
from .src.anchor_target_layer import AnchorTargetLayer


class RegionProposalNetwork(nn.Module):
    '''
    This is the whole region proposal network, which contains:
        1) feature extractor: basestone(!NOTE: just VGG16 supported now)
        2) proposal layer: to get rois and shifted anchors
        3) anchor terget layer: to get ground truth locs and ground truth labels
    '''
    def __init__(self, in_channel=512, mid_channel=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params = dict()):
        super(RegionProposalNetwork,self).__init__()

        self.feat_stride = feat_stride
        # define the featrue extractor
        self.extractor = basestone()
        # define the conv layer to processing input feature map
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1, bias=True)
        # generate basic anchors
        self.anchor_base = generate_anchor_base.chenyun_method(ratios=ratios, anchor_scales=anchor_scales)
        self.n_anchor = self.anchor_base.shape[0]
        # define bg/fg classificatinon score layer
        self.score = nn.Conv2d(mid_channel, self.n_anchor * 2, 1, 1, 0) # 2(bg/fg) * 9 (anchors)
        # define anchor box offset prediction layer
        self.loc = nn.Conv2d(mid_channel, self.n_anchor * 4, 1, 1, 0) # 4(coords) * 9 (anchors)
        # define proposal layer
        self.proposal_layer = ProposalLayer(self, **proposal_creator_params)
        # define anchor target layer
        self.anchor_target_layer = AnchorTargetLayer(n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5)
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

    def forward(self, imgs, gt_bbox, scale=1.,rpn_only=True):
        '''
        Args: 
             scale: default minimum size=16, need to ues scale to adjust min_size: min_size*scale;
        '''
        # get feature map from the basic extractor
        base_feature = self.extractor(imgs)
        # input the first conv(3X3)
        begin_feature = F.relu(self.conv1(base_feature))
        # forward the bg/fg classifier layer
        rpn_scores = self.score(begin_feature)
        rpn_scores_reshape = self.reshape(rpn_scores, 2)    # from (n,2*9,h,w) to (n,2,9*h,w)
        rpn_scores_pred_reshape = F.softmax(rpn_scores_reshape, 1)
        rpn_scores_pred = self.reshape(rpn_scores_pred_reshape, rpn_scores.size()[1])   # from (n,2,9*h,w) to (n,2*9,h,w)
        # forward the bounding box regression layer: get offsets to the anchor boxes: (n,4*9,h,w)
        rpn_locs_pred = self.loc(begin_feature)
        # forward proposal layer to get rois and anchors shifted on the image
        batch_size, _, hh, ww = base_feature.shape
        feature_shape = (hh, ww)
        imgs_size = imgs.shape[2:]
        rois, anchor = self.proposal_layer(rpn_locs_pred.cpu().data, rpn_scores_pred.cpu().data, self.anchor_base, 
                                           batch_size, feature_shape, imgs_size, scale.numpy())
        if rpn_only:
            # get ground truth bounfing box regression coefficients as well as labels
            # !NOTE: notice the shape of image, here should =3
            gt_locs, gt_labels = self.anchor_target_layer(gt_bbox.cpu(), anchor.cpu(), imgs_size)

            # Transpose and reshape predicted bbox transformations to get them
            # into the same order as the anchors(batch_size, K*9, 4); 
            # the same processing to the original rpn_score(no passing sofmax()) for calculating loss;
            rpn_locs_pred = rpn_locs_pred.permute(0,2,3,1).contiguous().reshape(batch_size, -1, 4)
            rpn_scores_pred = rpn_scores_reshape.permute(0,2,3,1).contiguous().reshape(batch_size,-1,2)
            return rpn_locs_pred, rpn_scores_pred, gt_locs, gt_labels
        else:
            return rois

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