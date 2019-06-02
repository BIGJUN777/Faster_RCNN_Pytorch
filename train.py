import numpy as np
import torch
from torch import optim
from utils.config import args
from model.rpn import RegionProposalNetwork
from data_tools.dataset import Dataset, TestDataset, inverse_normalize
import torch.nn.functional as F

from tqdm import tqdm
import ipdb

def train():

    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print('data prepared')
    faster_rcnn_rpn = RegionProposalNetwork()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    faster_rcnn_rpn.to(device)
    print('model construct completed')
    # ipdb.set_trace()
    # input = iter(dataloader).next()
    # test = faster_rcnn_rpn(input[0], input[1], input[3])
    optimizer = optim.Adam(faster_rcnn_rpn.parameters(), lr=1e-3)
    for epoch in range(args.epoch):
        for ii , (img, bbox, label, scale) in tqdm(enumerate(dataloader)):
            faster_rcnn_rpn.train()
            faster_rcnn_rpn.zero_grad()

            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            # ipdb.set_trace()
            rpn_locs_pred, rpn_scores_pred, gt_locs, gt_labels = faster_rcnn_rpn(img, bbox, scale)
            # Calculate the loss
            rpn_locs_pred.cuda()
            
            gt_label = torch.from_numpy(gt_labels).cuda().long()
            gt_loc = torch.from_numpy(gt_locs).cuda()
            rpn_cls_loss = F.cross_entropy(rpn_scores_pred.view(-1,2), gt_label, ignore_index=-1)
            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_locs_pred.view(-1,4), gt_loc, gt_label, args.rpn_sigma)
            rpn_loss = rpn_cls_loss + rpn_loc_loss
            rpn_loss.backward()
            optimizer.step()

            if ii % 100 == 0:
                print("rpn_cls_loss: {0}, rpn_loc_loss: {1}, rpn_loss: {2}".format(rpn_cls_loss, rpn_loc_loss, rpn_loss))

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