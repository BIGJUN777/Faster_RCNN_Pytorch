import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torch


def Faster_RCNN(num_classes):

    anchor_genreator = AnchorGenerator(sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0))
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                                 min_size=600, 
                                                                 max_size=1000,
                                                                #  rpn_anchor_generator=anchor_genreator,
                                                                 rpn_pre_nms_top_n_train=12000, 
                                                                 rpn_pre_nms_top_n_test=6000,
                                                                 rpn_post_nms_top_n_train=2000, 
                                                                 rpn_post_nms_top_n_test=300,
                                                                 rpn_nms_thresh=0.7,
                                                                 rpn_fg_iou_thresh=0.7, 
                                                                 rpn_bg_iou_thresh=0.3,
                                                                 rpn_batch_size_per_image=256, 
                                                                 rpn_positive_fraction=0.5,
                                                                 box_batch_size_per_image=128, 
                                                                 box_positive_fraction=0.25,
                                                                 box_score_thresh=0.1, 
                                                                 box_nms_thresh=0.5, 
                                                                 box_detections_per_img=100)

    model.rpn.anchor_genreator = anchor_genreator
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

@torch.no_grad()
def evalute(model, images):
    model.eval()
    outputs = model(images)
    outputs = [{k: v for k, v in t.items()} for t in outputs]
    return outputs

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    import numpy as np
    from PIL import Image
    from utils.vis_tool import vis_img

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # image = Image.open('database/000001.jpg').convert('RGB')
    # image.show()
    # image = np.array(image, dtype=np.float32).transpose((2,0,1))
    # input = torch.from_numpy(image)
    model.eval()
    image = Image.open('database/demo.jpg')
    image.show()
    input = torchvision.transforms.functional.to_tensor(image)
    outputs = model([input])
    img = vis_img(np.array(image), outputs[0]['boxes'].cpu().numpy(), outputs[0]['labels'].cpu().numpy(), outputs[0]['scores'].cpu().numpy())
    print(outputs)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]