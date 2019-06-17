import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from model.faster_rcnn import FasterRCNN
from utils.config import args
from data_tools.dataset import Dataset, TestDataset, inverse_normalize

from tqdm import tqdm
import ipdb

def train():
    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print('data prepared')
    model = FasterRCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('model construct completed')
    # ipdb.set_trace()
    # input = iter(dataloader).next()
    # test = model(input[0], input[1], input[3])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epoch):
        for ii , (img, bbox, label, scale) in tqdm(enumerate(dataloader)):
            model.train()
            model.zero_grad()

            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            ipdb.set_trace()
            rpn_cls_locs, rpn_scores, \
            gt_locs, gt_labels, \
            roi_cls_locs, roi_scores, \
            gt_roi_loc, gt_roi_label = model(img, bbox, label, scale)

            # ------------------ RPN losses -------------------#
            rpn_cls_locs.cuda()
            gt_labels = torch.from_numpy(gt_labels).cuda().long()
            gt_locs = torch.from_numpy(gt_locs).cuda()
            rpn_cls_loss = F.cross_entropy(rpn_scores.view(-1,2), gt_labels, ignore_index=-1)
            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_cls_locs.view(-1,4), gt_locs, gt_labels, args.rpn_sigma)
            

            # ------------------ ROI losses (fast rcnn loss) -------------------#
            n_simple = roi_cls_locs.shape[0]
            roi_cls_locs = roi_cls_locs.view((n_simple, -1 ,4))
            roi_loc = roi_cls_locs[np.arange(0,n_simple), gt_roi_label]
            gt_roi_label = torch.from_numpy(gt_roi_label).long().cuda()
            gt_roi_loc = torch.from_numpy(gt_roi_loc).cuda()
            roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_label.cuda())
            roi_loc_loss = _fast_rcnn_loc_loss(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, args.roi_sigma)

            loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
            loss.backward()
            optimizer.step()

            if ii % 100 == 0:
                print("roi_cls_loss: {0};\n roi_loc_loss: {1};\n rpn_cls_loss: {2};\n rpn_loc_loss: {3};\n loss: {4}\n" \
                        .format(roi_cls_loss, roi_loc_loss, rpn_cls_loss, rpn_loc_loss, loss))

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
    
if __name__ == "__main__":
    train()
