import argparse

def argument():
    '''
    set up configumation
    '''
    parse = argparse.ArgumentParser(description="Train a Faster R-CNN network!")

    parse.add_argument('--net', type=str,
                        help="choose which CNN as basestone: vgg16",
                        choices=['vgg16', 'res101'],
                        default='vgg16')
    parse.add_argument('--caffe_pretrain',type=bool,
                        help="pretrain vgg in caffe: False",
                        default=False)
    parse.add_argument('--use_drop', type=bool,
                        help="use drop layers in classifier: True",
                        default=True)
    parse.add_argument('--nms_thresh', type=float,
                        help='the thresh of non maximun suppression: 0.7',
                        default=0.7)
    parse.add_argument('--n_pre_nms_train', type=int,
                        help='select roi with the top N score before nms: 12000',
                        default=12000)
    parse.add_argument('--n_post_nms_train', type=int,
                        help='select roi with the top N score after nms: 2000',
                        default=12000)
    parse.add_argument('--n_pre_nms_test', type=int,
                        help='select roi with the top N score before nms: 6000',
                        default=6000)
    parse.add_argument('--n_post_nms_test', type=int,
                        help='select roi with the top N score after nms: 300',
                        default=300)
    parse.add_argument('--min_size', type=int,
                        help='set the minimum heigth or width size of : 16',
                        default=16)
    parse.add_argument('--n_simple', type=int,
                        help='the number of the gt regions to produce : 256',
                        default=256)
    parse.add_argument('--pos_iou_thresh', type=float,
                        help='Anchors with IoU above this threshold will be assigned as positive : 0.7',
                        default=0.7)
    parse.add_argument('--neg_iou_thresh', type=float,
                        help='Anchors with IoU above this threshold will be assigned as positive : 0.3',
                        default=0.3)
    parse.add_argument('--voc_data_dir',type=str,
                        help='path to database',
                        default='database/VOCdevkit2007/VOC2007')
    parse.add_argument('--min_img_size', type=int,
                        help='the number of the gt regions to produce : 256',
                        default=600)
    parse.add_argument('--max_img_size', type=int,
                        help='the number of the gt regions to produce : 256',
                        default=1000)
    parse.add_argument('--epoch', type=int,
                        help='the number of the gt regions to produce : 256',
                        default=100)
    parse.add_argument('--rpn_sigma', type=int,
                        help='sigma for l1 smooth_loss : 3.',
                        default=3.)
    parse.add_argument('--roi_sigma', type=int,
                        help='sigma for l1 smooth_loss : 3.',
                        default=1.)

    return parse.parse_args()

args = argument()