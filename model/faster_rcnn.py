import torch
import torch.nn as nn
from torch.nn import functional as F
from .basestone import BaseStone 
from .rpn import RegionProposalNetwork
from .src.anchor_target_layer import AnchorTargetLayer
from .src.proposal_target_layer import ProposalTargetCreator
from .src.roi_pooling_layer import RoIPooling2D
import ipdb

class FasterRCNN(nn.Module):
    '''
    The whole faster RCNN model, which including:

    '''
    def __init__(self, net='vgg16', feat_stride=16, n_class=21):
        '''
        n_class: 20 + 1(background)
        '''
        super(FasterRCNN, self).__init__()

        _base = BaseStone()
        # define extractor and classifier
        self.extractor, self.classifier = _base(net)
        # define layers after extractor till proposal layer
        self.rpn = RegionProposalNetwork(in_channel=512, mid_channel=512, ratios=[0.5, 1, 2],
                                         anchor_scales=[8, 16, 32], feat_stride=16,)
        # define anchor target layer
        self.anchor_target_layer = AnchorTargetLayer(n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5)
        # define proposal target layer
        self.proposal_target_layer = ProposalTargetCreator(n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5,
                                                           neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.1)
        # define roipooling layer
        self.roi_pooling_layer = RoIPooling2D(7, 7, 1./feat_stride)
        # define last linear layers
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

    def forward(self, img, gt_bbox, label, scale):

        # ipdb.set_trace()
        img_size = img.shape[2:]
        # get feature map from the basic extractor
        base_feature = self.extractor(img)
        # forword the rpn
        rois, anchor, rpn_cls_locs, rpn_scores = self.rpn(base_feature, gt_bbox, img_size, scale)
        if self.training:
            # get ground truth bounfing box regression coefficients as well as labels
            gt_locs, gt_labels = self.anchor_target_layer(gt_bbox.cpu().numpy(), anchor.cpu().numpy(), img_size)
        # forward the proposal target layer
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(rois, gt_bbox[0].cpu().numpy(), label[0].cpu().numpy())
        # forward the roi pooling layer
        # !NOTE: need to clean more
        sample_roi_index = torch.zeros(len(sample_roi))
        pool = self.roi_pooling_layer(base_feature, sample_roi, sample_roi_index)
        # forward the last layers
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        if self.training:
            return rpn_cls_locs, rpn_scores, gt_locs, gt_labels, roi_cls_locs, roi_scores, gt_roi_loc, gt_roi_label
        else:
            return roi_cls_locs, roi_scores, gt_roi_loc, gt_roi_label

