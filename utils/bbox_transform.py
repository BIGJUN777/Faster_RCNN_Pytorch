import numpy as np
import numpy as xp
import torch

def loc2bbox(boxes, deltas):
    '''
    Decode bounding boxes from bounding box offsets and scales
    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`
    '''
    widths = boxes[:, :, 2] - boxes[:, :, 0] 
    heights = boxes[:, :, 3] - boxes[:, :, 1] 
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dh = deltas[:, :, 2::4]
    dw = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox2loc(src_bbox, dst_bbox):
    """
    Encodes the source and the destination bounding boxes to "loc".
    Args:  
         src_bbox: (batch_size, ..., 4)

    """

    width = src_bbox[ :, 2] - src_bbox[ :, 0]
    height = src_bbox[ :, 3] - src_bbox[ :, 1]
    ctr_x = src_bbox[ :, 0] + 0.5 * width
    ctr_y = src_bbox[ :, 1] + 0.5 * height

    base_width = dst_bbox[ :, 2] - dst_bbox[ :, 0]
    base_height = dst_bbox[ :, 3] - dst_bbox[ :, 1]
    base_ctr_x = dst_bbox[ :, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[ :, 1] + 0.5 * base_height

    # eps = xp.finfo(height.dtype).eps
    # height = xp.maximum(height, eps)
    # width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dx, dy, dh, dw)).transpose()
    return loc

def clip_boxes(boxes, im_shape, batch_size):
    '''
    !NOTE im_shape ---> (height, width)
    '''
    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[1])
        boxes[i,:,1::4].clamp_(0, im_shape[0])
        boxes[i,:,2::4].clamp_(0, im_shape[1])
        boxes[i,:,3::4].clamp_(0, im_shape[0])

    return boxes


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        !NOTE: add batch_size---> (batch_size, N,4)
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    
    bbox_a = bbox_a.reshape((-1,4))
    bbox_b = bbox_b.reshape((-1,4))
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

    # """
    # anchors: (N, 4) ndarray of float
    # gt_boxes: (K, 4) ndarray of float
    # overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    # """
    # N = anchors.size(0)
    # K = gt_boxes.size(0)

    # gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
    #             (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    # anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
    #             (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    # boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    # query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    # iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
    #     torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    # iw[iw < 0] = 0

    # ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
    #     torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    # ih[ih < 0] = 0

    # ua = anchors_area + gt_boxes_area - (iw * ih)
    # overlaps = iw * ih / ua

    # return overlaps