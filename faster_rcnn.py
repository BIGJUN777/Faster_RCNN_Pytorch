import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch



def Faster_RCNN(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

@torch.no_grad()
def evalute(model, images):
    model.eval()
    outputs = model(images)
    outputs = [{k: v for k, v in t.items()} for t in outputs]
    return outputs


def vis_img(img, bboxs, labels, scores=None):
    try:
        if len(bboxs) == 0:
            return img    

        if scores is not None:
            keep = np.where(scores > 0.8)
            bboxs = bboxs[keep]
            labels = labels[keep]
            scores = scores[keep] 
        
        score_idx = 0 
        line_width = 3
        for (bbox, label) in zip(bboxs, labels):
            Drawer = ImageDraw.Draw(img)
            # ipdb.set_trace()
            Drawer.rectangle(list(bbox), outline='red', width=line_width)
            text = COCO_INSTANCE_CATEGORY_NAMES[label]
            if scores is None:
                Drawer.text((bbox[0]+line_width+1, bbox[1]+line_width+1), text, 'red')
            else:
                text = text + " " + '{:.3f}'.format(scores[score_idx])
                Drawer.text((bbox[0]+line_width+1, bbox[1]+line_width+1), text, 'red' )
                score_idx +=1
        return img

    except Exception as e:
        print("Error:" ,e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass


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

if __name__ == "__main__":
    import numpy as np
    from tensorboardX import SummaryWriter
    from PIL import Image, ImageDraw
    import time

    writer = SummaryWriter(log_dir='log')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # image = Image.open('database/000001.jpg').convert('RGB')
    # image.show()
    # image = np.array(image, dtype=np.float32).transpose((2,0,1))
    # input = torch.from_numpy(image)
    model.eval()
    image = Image.open('database/demo.jpg').convert('RGB')
    image.show()
    input = torchvision.transforms.functional.to_tensor(image)
    outputs = model([input])
    img = vis_img(image, outputs[0]['boxes'].detach().numpy(), outputs[0]['labels'].detach().numpy(), outputs[0]['scores'].detach().numpy())
    while True:
        writer.add_image('test', np.array(img).transpose(2,0,1))
        time.sleep(5)
    writer.close()
    print(outputs)

