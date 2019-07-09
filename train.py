import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torchvision
# from utils.eval_tool import eval_detection_voc
from utils.vis_tool import vis_img
# from data_tools.dataset import Dataset, TestDataset, inverse_normalize
from data_tools.dataset import inverse_normalize
from data_tools.dataset import get_voc
from model.faster_rcnn import Faster_RCNN, evalute
from utils.utils import collate_fn

import os
import ipdb
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse


def run_network(args):

    # dataset = Dataset(data_dir='database/VOCdevkit2007/VOC2007')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # testset = TestDataset(data_dir='database/VOCdevkit2007/VOC2007')
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # print('data prepared, train data: %d, test data: %d' % (len(dataset), len(testset)))

    dataset = get_voc('database', 'trainval', )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print('data prepared, train data: %d' % len(dataset))
    model = Faster_RCNN(num_classes=21)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    model.to(device)
    print('model construct completed. Training on %s' % device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    # optimizer = optim.Adam(model.parameters(), lr=0.005)
    optimizer = optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # setup log data writer
    if not os.path.exists('log'):
        os.makedirs('log')
    writer = SummaryWriter(log_dir='log')

    for epoch in range(300):
        loss_epoch = {}
        loss_name = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
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
            # ipdb.set_trace()
            if ii == 0:
                for name in loss_name:
                    loss_epoch[name] = loss_dict[name].item() * args.batch_size
            else:
                for name in loss_name:
                    loss_epoch[name] += loss_dict[name].item() * args.batch_size

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

        for k, v in loss_epoch.items():
            loss_epoch[k] = v /len(dataset)

        writer.add_scalars('loss', loss_epoch, epoch)
        lr_scheduler.step()

        # save checkpoints
        if not os.path.exists('./checkpoints'):
            os.makedirs("./checkpoints")
        if epoch % 100 == 99:
            torch.save(model.state_dict(), './checkpoints/checkpoint_{}_epochs.pth'.format(epoch+1))

if __name__ == "__main__":

    def str2bool(arg):
        arg = arg.lower()
        if arg in ['true', '1', 'yes']:
            return True
        elif arg in ['false', '0', 'no']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected!')

    parse = argparse.ArgumentParser(description="faster-rcnn")
    parse.add_argument('--batch_size', type=int, default=1,
                        help="batch_size: 1")
    
    parse.add_argument('--gpu', type=str2bool, default='true',
                        help="use gpu or not: true")

    args = parse.parse_args()

    run_network(args)
