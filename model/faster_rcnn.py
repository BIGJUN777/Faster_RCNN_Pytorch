import torch
import torch.nn as nn
from torch.nn import functional as F
from .basestone import BaseStone 
from .rpn import RegionProposalNetwork
from .src.anchor_target_layer import AnchorTargetLayer
from .src.proposal_target_layer import ProposalTargetCreator
from .src.roi_pooling_layer import RoIPooling2D
from utils.nms.non_maximum_suppression import non_maximum_suppression
from data_tools.dataset import preprocess
from utils import array_tool as at
import cupy as cp
import numpy as np
import ipdb

def nograd(f):
    def new_f(*args,**kwargs):
        with torch.no_grad():
           return f(*args,**kwargs)
    return new_f

class FasterRCNN(nn.Module):
    '''
    The whole faster RCNN model, which including:

    '''
    def __init__(self, net='vgg16', feat_stride=16, n_class=21):
        '''
        n_class: 20 + 1(background)
        '''
        super(FasterRCNN, self).__init__()

        self.n_class = n_class
        _base = BaseStone()
        # define extractor and classifier
        self.extractor, self.classifier = _base(net)
        # define layers after extractor till proposal layer
        self.rpn = RegionProposalNetwork(in_channel=512, mid_channel=512, ratios=[0.5, 1, 2],
                                         anchor_scales=[8, 16, 32], feat_stride=16,)
        # define anchor target layer
        self.anchor_target_layer = AnchorTargetLayer(n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5)
        # define proposal target layer
        self.loc_normalize_mean=(0., 0., 0., 0.)
        self.loc_normalize_std=(0.1, 0.1, 0.2, 0.2)
        self.proposal_target_layer = ProposalTargetCreator(n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5,
                                                           neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.1)
        # define roipooling layer
        self.roi_pooling_layer = RoIPooling2D(7, 7, 1./feat_stride)
        # define last linear layers
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

    def forward(self, img=None, gt_bbox=None, label=None, scale=None):

        # ipdb.set_trace()
        img_size = img.shape[2:]
        # get feature map from the basic extractor
        base_feature = self.extractor(img)
        # forword the rpn
        rois, anchor, rpn_cls_locs, rpn_scores = self.rpn(base_feature, gt_bbox, img_size, scale)
        if self.training:
            # anchor_target_layer and proposal_target_layer just are used in train phase
            # get ground truth bounfing box regression coefficients as well as labels
            gt_locs, gt_labels = self.anchor_target_layer(gt_bbox.cpu().numpy(), anchor.cpu().numpy(), img_size)
            # forward the proposal target layer
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_layer(rois, gt_bbox[0].cpu().numpy(), label[0].cpu().numpy(), self.loc_normalize_mean, self.loc_normalize_std)
        # forward the roi pooling layer
        # !NOTE: need to clean more
        else:
            sample_roi = rois.numpy()
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
            return roi_cls_locs, roi_scores, rois

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`self.nms_thresh` and
        :obj:`self.score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'): A string to determine the
                preset to use.

        """
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @staticmethod
    def loc2bbox(src_bbox, loc):
        """Decode bounding boxes from bounding box offsets and scales.

        Given bounding box offsets and scales computed by
        :meth:`bbox2loc`, this function decodes the representation to
        coordinates in 2D image coordinates.

        Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
        box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
        the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
        and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
        by the following formulas.

        * :math:`\\hat{g}_y = p_h t_y + p_y`
        * :math:`\\hat{g}_x = p_w t_x + p_x`
        * :math:`\\hat{g}_h = p_h \\enp(t_h)`
        * :math:`\\hat{g}_w = p_w \\enp(t_w)`

        The decoding formulas are used in works such as R-CNN [#]_.

        The output is same type as the type of the inputs.

        .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
        Rich feature hierarchies for accurate object detection and semantic \
        segmentation. CVPR 2014.

        Args:
            src_bbox (array): A coordinates of bounding boxes.
                Its shape is :math:`(R, 4)`. These coordinates are
                :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
            loc (array): An array with offsets and scales.
                The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
                This contains values :math:`t_y, t_x, t_h, t_w`.

        Returns:
            array:
            Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
            The second axis contains four values \
            :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
            \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

        """

        if src_bbox.shape[0] == 0:
            return np.zeros((0, 4), dtype=loc.dtype)

        src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

        src_height = src_bbox[:, 2] - src_bbox[:, 0]
        src_width = src_bbox[:, 3] - src_bbox[:, 1]
        src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
        src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

        dy = loc[:, 0::4]
        dx = loc[:, 1::4]
        dh = loc[:, 2::4]
        dw = loc[:, 3::4]

        ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
        ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
        h = np.exp(dh) * src_height[:, np.newaxis]
        w = np.exp(dw) * src_width[:, np.newaxis]

        dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
        dst_bbox[:, 0::4] = ctr_y - 0.5 * h
        dst_bbox[:, 1::4] = ctr_x - 0.5 * w
        dst_bbox[:, 2::4] = ctr_y + 0.5 * h
        dst_bbox[:, 3::4] = ctr_x + 0.5 * w

        return dst_bbox

    @nograd
    def predict(self, imgs,sizes=None,visualize=False):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bouding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs 
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            scale = np.array(scale)
            roi_cls_loc, roi_scores, rois = self(img, scale=scale)
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            # ipdb.set_trace()
            roi = rois / float(scale)
            
            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = torch.Tensor(self.loc_normalize_mean * self.n_class).cuda()
            std = torch.Tensor(self.loc_normalize_std * self.n_class).cuda()

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)

            roi = roi.view(-1, 1, 4).repeat(1,roi_cls_loc.shape[1],1)
            cls_bbox = self.loc2bbox(at.tonumpy(roi).reshape(-1,4),
                                at.tonumpy(roi_cls_loc).reshape(-1,4))

            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[1])

            prob = at.tonumpy(F.softmax(at.totensor(roi_score), dim=1))

            raw_cls_bbox = at.tonumpy(cls_bbox)
            raw_prob = at.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores