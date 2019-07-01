# code based on chenyun
# https://github.com/chenyuntc/simple-faster-rcnn-pytorch

# import sys
# sys.path.append('..')
import torch 
import torch.nn as nn
from torchvision import models
from utils.config import args

class BaseStone(nn.Module):
    '''
    Faster R-CNN(FRRC) is based on popular CNN structure like: VGG; Resnet.
    This class is aimed to bulid the extractor as the begining structure of FRRC
    and classifier as the last structure of FRRC.
    '''
    def __init__(self, net='vgg16'):
        super(BaseStone, self).__init__()
        if net == 'vgg16':
            self.extractor, self.classifier = _decompose_vgg16()
        else:
            self.extractor, self.classifier = _decompose_resnet()
    def forward(self, net ='vgg16'):
        return self.extractor, self.classifier

def _decompose_vgg16():
    '''
    decompose vgg16 into extractor and classifier
    '''
    # use pretrain caffe's model or not 
    if args.caffe_pretrain:
        model = models.vgg16(not args.caffe_pretrain)
        model.load_state_dict(torch.load())
    else:
        model = models.vgg16(not args.caffe_pretrain)
    # delete the last MaxPooling layer
    # freeze the first 4 convs
    features = list(model.features)[:30]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    features = nn.Sequential(*features)
    # delete the last Linear layer
    classifier = list(model.classifier)
    del classifier[6]
    if not args.use_drop:
        del classifier[2]
        del classifier[5]
    classifier = nn.Sequential(*classifier)

    return features, classifier

def _decompose_resnet():
    pass