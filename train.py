import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from model.faster_rcnn import FasterRCNN
from utils.config import args
from utils.eval_tool import eval_detection_voc
from utils.vis_tool import vis_img
from data_tools.dataset import Dataset, TestDataset, inverse_normalize

from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import cv2
import ipdb


def run_network():
    dataset = Dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    testset = TestDataset(args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    print('data prepared, train data: %d, test data: %d' % (len(dataset), len(testset)))
    model = FasterRCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('model construct completed')
    # ipdb.set_trace()
    # input = iter(dataloader).next()
    # test = model(input[0], input[1], input[3])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # setup log data writer
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(args.epoch):
        train_losses = [0] * 5
        for ii , (img, bbox, label, scale) in tqdm(enumerate(dataloader)):
            # img:(batch_size, 3, h, w), bbox:(batch_size, n, 4), label:(batch, n), scale:(1)
            model.train()
            model.zero_grad()

            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            # ipdb.set_trace()
            rpn_cls_locs, rpn_scores, \
            gt_locs, gt_labels, \
            roi_cls_locs, roi_scores, \
            gt_roi_loc, gt_roi_label = model(img, bbox, label, scale.numpy())

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

            # save loss
            train_losses[0] += loss.item()
            train_losses[1] += rpn_loc_loss.item()
            train_losses[2] += rpn_cls_loss.item()
            train_losses[3] += roi_loc_loss.item()
            train_losses[4] += roi_cls_loss.item()
            # plot image
            if (ii == 0) or ((ii+1) % args.plot_every) == 0:
                # ipdb.set_trace()
                ori_img = inverse_normalize(img[0].cpu().numpy())
                # plot original image
                img = vis_img(ori_img, bbox[0].cpu().numpy(), label[0].cpu().numpy())
                writer.add_image('original_images', np.array(img).transpose(2,0,1), ii)
                # plot pred image
                _bboxes, _labels, _scores = model.predict([ori_img], visualize=True)
                if not len(_bboxes[0]) == 0:
                    print("pred_bboxes: {}, pre_labels: {}, pre_score: {}".format(_bboxes, _labels, _scores))
                img = vis_img(ori_img, _bboxes[0], _labels[0].reshape(-1), _scores[0].reshape(-1))
                writer.add_image('pre_images', np.array(img).transpose(2,0,1), ii)
            # if ii == 2 : break
                # print("EPOCH:[{}/{}], roi_cls_loss: {};\n roi_loc_loss: {};\n rpn_cls_loss: {};\n rpn_loc_loss: {};\n loss: {}\n" \
                #         .format(epoch, args.epoch, roi_cls_loss, roi_loc_loss, rpn_cls_loss, rpn_loc_loss, loss))

        # testing
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(testloader)):
            model.eval()
            with torch.no_grad():
                sizes = [sizes[0][0].item(), sizes[1][0].item()]
                pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs.to(device), [sizes])
                gt_bboxes += list(gt_bboxes_.numpy())
                gt_labels += list(gt_labels_.numpy())
                gt_difficults += list(gt_difficults_.numpy())
                pred_bboxes += pred_bboxes_
                pred_labels += pred_labels_
                pred_scores += pred_scores_
            # if ii == 500: break

        test_result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)
        # plot loss and mAP
        # ipdb.set_trace()
        loss_name = ['total_loss', 'rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss']
        values = {}
        for i in range(len(loss_name)):
            values[loss_name[i]] = train_losses[i] / len(dataset)
        writer.add_scalars('loss', values, epoch)
        writer.add_scalar('test_mAP', test_result['map'], epoch)

        # save checkpoints
        if not os.path.exists('./checkpoints'):
            os.makedirs("./checkpoints")
        if epoch % 100 == 99:
            torch.save(model.state_dict(), './checkpoints/checkpoint_{}_epochs.pth'.format(epoch))

            # plot images
            # if (ii+1) % args.plot_every == 0:

            #     print("roi_cls_loss: {0};\n roi_loc_loss: {1};\n rpn_cls_loss: {2};\n rpn_loc_loss: {3};\n loss: {4}\n" \
            #             .format(roi_cls_loss, roi_loc_loss, rpn_cls_loss, rpn_loc_loss, loss))
        

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
    run_network()
