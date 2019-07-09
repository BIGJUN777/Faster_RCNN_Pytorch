# from __future__ import  absolute_import
# from __future__ import  division
# import torch as t
# from data_tools.voc_dataset import VOCBboxDataset
# from data_tools import util
# from skimage import transform as sktsf
# from torchvision import transforms as tvtsf
# import numpy as np
# import torch
# import ipdb

def inverse_normalize(img, caffe_pretrain=False):
    if caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


# def pytorch_normalze(img):
#     """
#     https://github.com/pytorch/vision/issues/223
#     return appr -1~1 RGB
#     """
#     normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#     img = normalize((t.from_numpy(img)).float())
#     return img.numpy()

# def caffe_normalize(img):
#     """
#     return appr -125-125 BGR
#     """
#     img = img[[2, 1, 0], :, :]  # RGB-BGR
#     img = img * 255
#     mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
#     img = (img - mean).astype(np.float32, copy=True)
#     return img


# # def preprocess(img, min_size=600, max_size=1000):
# #     """Preprocess an image for feature extraction.

# #     The length of the shorter edge is scaled to :obj:`self.min_size`.
# #     After the scaling, if the length of the longer edge is longer than
# #     :param min_size:
# #     :obj:`self.max_size`, the image is scaled to fit the longer edge
# #     to :obj:`self.max_size`.

# #     After resizing the image, the image is subtracted by a mean image value
# #     :obj:`self.mean`.

# #     Args:
# #         img (~numpy.ndarray): An image. This is in CHW and RGB format.
# #             The range of its value is :math:`[0, 255]`.

# #     Returns:
# #         ~numpy.ndarray: A preprocessed image.

# #     """
# #     C, H, W = img.shape
# #     scale1 = min_size / min(H, W)
# #     scale2 = max_size / max(H, W)
# #     scale = min(scale1, scale2)
# #     img = img / 255.
# #     img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
# #     # both the longer and shorter should be less than
# #     # max_size and min_size
# #     if args.caffe_pretrain:
# #         normalize = caffe_normalize
# #     else:
# #         normalize = pytorch_normalze
# #     return normalize(img)

# class Transform(object):

#     def __init__(self, min_size=600, max_size=1000):
#         self.min_size = min_size
#         self.max_size = max_size

#     def __call__(self, in_data):
#         img, bbox, label = in_data
#         img = img / 255
#         _, o_H, o_W = img.shape

#         # horizontally flip
#         img, params = util.random_flip(
#             img, x_random=True, return_param=True)
#         bbox = util.flip_bbox(
#             bbox, (o_H, o_W), x_flip=params['x_flip'])

#         return img, bbox, label

# class Dataset:
#     def __init__(self, data_dir, min_size=600, max_size=1000):
#         self.db = VOCBboxDataset(data_dir)
#         self.tsf = Transform(min_size, max_size)

#     def __getitem__(self, idx):
#         # get raw data from original database 
#         ori_img, bbox, label, difficult = self.db.get_example(idx)
#         # pre-process data: resize
        
#         img, bbox, label = self.tsf((ori_img, bbox, label))
#         # convert everything into a torch.Tensor
#         bbox = torch.as_tensor(bbox, dtype=torch.float32)
#         label = torch.as_tensor(label, dtype=torch.int64)
#         img = torch.from_numpy(img.clip(min=0, max=1))
#         # ipdb.set_trace()
#         target = {}
#         target['boxes'] = bbox
#         target['labels'] = label
#         return img, target

#     def __len__(self):
#         return len(self.db)


# class TestDataset:
#     def __init__(self, data_dir, split='test', use_difficult=True):
#         self.db = VOCBboxDataset(data_dir, split=split, use_difficult=use_difficult)

#     def __getitem__(self, idx):
#         ori_img, bbox, label, difficult = self.db.get_example(idx)
#         img = ori_img / 255
#         img = torch.from_numpy(img.clip(min=0, max=1))
#         target = {}
#         target['boxes'] = bbox
#         target['labels'] = label
#         return img, target

#     def __len__(self):
#         return len(self.db)

# if __name__ == "__main__":
#     import sys
#     sys.path.append("..")
#     import cv2
#     trainloader = Dataset(args)
#     testloader = TestDataset(args)
#     train = iter(trainloader)
#     test = iter(testloader)
#     import ipdb; ipdb.set_trace()
#     img, bbox, label, scale = next(train)
#     ori_img = inverse_normalize(img[0].numpy())
#     cv2.imshow('img', img[0].cpu().numpy().transpose(1,2,0))
#     cv2.waitKey(100)


import torch
import torchvision

import utils.transforms as T


class PrepareInstance(object):
    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    def __call__(self, image, target):
        anno = target['annotation']
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            iscrowd.append(int(obj['difficult']))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = anno['filename'][5:-4]
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def get_voc(root, image_set, transforms=None):
    t = [PrepareInstance()]

    if transforms is not None:
        t.append(transforms)
    t.append(T.ToTensor())
    t.append(T.RandomHorizontalFlip(0.5))
    transforms = T.Compose(t)

    dataset = torchvision.datasets.VOCDetection(root, '2007', image_set, transforms=transforms, download=False)

    return dataset