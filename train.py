import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
# from utils.eval_tool import eval_detection_voc
from utils.vis_tool import vis_img
from data_tools.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn import Faster_RCNN, evalute

import os
import cv2
import ipdb
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.utils import collate_fn
import torchvision

def run_network():
    dataset = Dataset(data_dir='database/VOCdevkit2007/VOC2007')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    testset = TestDataset(data_dir='database/VOCdevkit2007/VOC2007')
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print('data prepared, train data: %d, test data: %d' % (len(dataset), len(testset)))
    model = Faster_RCNN(num_classes=20)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('model construct completed')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # setup log data writer
    if not os.path.exists('log'):
        os.makedirs('log')
    writer = SummaryWriter(log_dir='log')

    for epoch in range(100):
        for ii, (images, targets) in tqdm(enumerate(dataloader)):   
            model.train()
            model.zero_grad()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            # print(loss_dicts)
            # plot image
            if (ii == 0) or ((ii+1) % 50) == 0:
                # ipdb.set_trace()
                ori_img = inverse_normalize(images[0].cpu().numpy())
                # plot original image
                img = vis_img(ori_img, targets[0]['boxes'].cpu().numpy(), targets[0]['labels'].cpu().numpy())
                writer.add_image('original_images', np.array(img).transpose(2,0,1), ii)
            
                # plot pred image
                outputs = evalute(model, images)
                if not len(outputs) == 0:
                    print("pred_bboxes: {}, pre_labels: {}, pre_score: {}".format(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']))
                img = vis_img(ori_img, outputs[0]['boxes'].cpu().numpy(), outputs[0]['labels'].cpu().numpy(), outputs[0]['scores'].cpu().numpy())
                writer.add_image('pre_images', np.array(img).transpose(2,0,1), ii)

            writer.add_scalars('loss', loss_dict, ii)
        lr_scheduler.step()
        # testing
        # for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(testloader)):
        #     model.eval()
        #     with torch.no_grad():
        #         sizes = [sizes[0][0].item(), sizes[1][0].item()]
        #         pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs.to(device), [sizes])
        #         gt_bboxes += list(gt_bboxes_.numpy())
        #         gt_labels += list(gt_labels_.numpy())
        #         gt_difficults += list(gt_difficults_.numpy())
        #         pred_bboxes += pred_bboxes_
        #         pred_labels += pred_labels_
        #         pred_scores += pred_scores_
        #     # if ii == 500: break

        # test_result = eval_detection_voc(
        #     pred_bboxes, pred_labels, pred_scores,
        #     gt_bboxes, gt_labels, gt_difficults,
        #     use_07_metric=True)

        # save checkpoints
        if not os.path.exists('./checkpoints'):
            os.makedirs("./checkpoints")
        if epoch % 100 == 99:
            torch.save(model.state_dict(), './checkpoints/checkpoint_{}_epochs.pth'.format(epoch))

if __name__ == "__main__":
    run_network()
