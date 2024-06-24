"""
numpy로 구성된 유틸
utils_graph와 부분적 기능이 동일한 함수로 구성
"""
import os
import numpy as np
import pickle
import time
from typing import List
# from ..common.teethConfigure import
# from .image_utils import get_interpolation
# from . import image_utils
from commons.teethConfigure import get_fdi2label
from commons.TeethStructure import TeethStructure
from commons import get_runtime_logger


# https://stackoverflow.com/questions/33945261/how-to-specify-multiple-return-types-using-type-hints
def search_dicom_directory(dirname, extensions=['.dcm', '.DCM'])-> List[str]:
    """
    directory-tree 일괄 탐색해서 dicom-files즈가 일정 개수(100)이상인 디렉토리를 가져온다.

    Args:
        dirname ():
        extensions ():

    Returns: List[int] the directory of dicom-files

    """

    dicom_dirs = []
    for root, dirs, files in os.walk(dirname):
        # print(root, dirs, len(files))
        res = [file.lower().endswith(tuple(extensions)) for file in files]
        # print(all(res))
        if len(res) > 100:
            # print(all(res), len(res))
            # print(root, len(res))
            dicom_dirs.append(root)
    return dicom_dirs

def get_deep_dicom_directory(dirname) -> str:
    res = search_dicom_directory(dirname)
    if res:
        return res[0]
    else:
        return []


def visualize_mask(volume, mask, alpha=0.5):
    import cv2
    import matplotlib.pyplot as plt
    for img, mask in zip(volume, mask):
        drawing = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        apply_mask(drawing, mask > .5, (0, 1, 0), alpha=alpha)
        plt.cla()
        plt.imshow(drawing)
        plt.pause(0.01)


def norm_boxes(boxes, shape):
    """
    likewise opencv
    OpenCV typically assumes that the top and left boundary of the rectangle are inclusive,
    while the right and bottom boundaries are not. For example, the method Rect_::contains returns true if
    x≤pt.x<x+width,y≤pt.y<y+height

    Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    d, h, w = shape
    scale = np.array([d - 1, h - 1, w - 1, d - 1,  h - 1, w - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def norm_boxes_bounds(boxes, shape):
    scale = np.asarray(shape) - 1
    scale = np.pad(scale, [0, 3], mode='wrap')
    # d, h, w = shape
    # scale = np.array([d - 1, h - 1, w - 1, d - 1,  h - 1, w - 1])
    # shift = np.array([0, 0, 0, 1, 1, 1]) #inner_crop
    return np.divide(boxes, scale).astype(np.float32)


def denorm_boxes_bounds(boxes, shape):
    scale = np.asarray(shape)-1
    scale = np.pad(scale, [0, 3], mode='wrap')
    return np.multiply(boxes, scale)

def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    """
    d, h, w = shape
    scale = np.array([d - 1, h - 1, w - 1, d - 1, h - 1, w - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (z1, y1, x1, y2, x2)] boxes to update
    deltas: [N, (dz, dy, dx, log(dd), log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]

    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= np.exp(deltas[:, 3])
    height *= np.exp(deltas[:, 4])
    width *= np.exp(deltas[:, 5])
    # Convert back to y1, x1, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    result = np.stack([z1, y1, x1, z2, y2, x2], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]

    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = np.log(gt_depth / depth)
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dz, dy, dx, dd, dh, dw], axis=1)
#
# def box_refinement(box, gt_box):
#     """Compute refinement needed to transform box to gt_box.
#     box and gt_box are [N, (z1, y1, x1, z2, y2, x2)] is
#     assumed to be outside the box.
#     """
#
#     box = box.astype(np.float32)
#     gt_box = gt_box.astype(np.float32)
#
#     depth = box[:, 3] - box[:, 0]
#     height = box[:, 4] - box[:, 1]
#     width = box[:, 5] - box[:, 2]
#
#     center_z = box[:, 0] + 0.5 * depth
#     center_y = box[:, 1] + 0.5 * height
#     center_x = box[:, 2] + 0.5 * width
#
#     gt_depth = gt_box[:, 3] - gt_box[:, 0]
#     gt_height = gt_box[:, 4] - gt_box[:, 1]
#     gt_width = gt_box[:, 5] - gt_box[:, 2]
#
#     gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
#     gt_center_y = gt_box[:, 1] + 0.5 * gt_height
#     gt_center_x = gt_box[:, 2] + 0.5 * gt_width
#
#     dz = (gt_center_z - center_z) / depth
#     dy = (gt_center_y - center_y) / height
#     dx = (gt_center_x - center_x) / width
#     dd = np.log(gt_depth / depth)
#     dh = np.log(gt_height / height)
#     dw = np.log(gt_width / width)
#
#     return np.stack([dz, dy, dx, dd, dh, dw], axis=1)



def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image




def apply_blending_mask(image_a, image_b, alpha=0.5):
    """Apply the given mask to the image. mask is computed non-zero value from image_b
    """
    assert image_a.shape == image_b.shape
    # mask =
    mask = np.any(image_b > 0, axis=-1)
    for c in range(3):
        image_a[:, :, c] = np.where(mask > 0,
                           image_a[:, :, c] * (1 - alpha) + \
                           image_b[:, :, c] * alpha, image_a[:, :, c])
    return image_a



############################################################
#  Detection Target Layer
############################################################

def compute_iou(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = np.reshape(np.tile(np.expand_dims(boxes1, 1),
                            [1, 1, np.shape(boxes2)[0]]), [-1, 6])
    b2 = np.tile(boxes2, [np.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = np.split(b1, 6, axis=1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = np.split(b2, 6, axis=1)
    z1 = np.maximum(b1_z1, b2_z1)
    y1 = np.maximum(b1_y1, b2_y1)
    x1 = np.maximum(b1_x1, b2_x1)
    z2 = np.minimum(b1_z2, b2_z2)
    y2 = np.minimum(b1_y2, b2_y2)
    x2 = np.minimum(b1_x2, b2_x2)
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    # 3. Compute unions
    b1_area = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = np.reshape(iou, [np.shape(boxes1)[0], np.shape(boxes2)[0]])
    return overlaps



# def compute_iou_graph(boxes1, boxes2):
#     """Computes IoU overlaps between two sets of boxes.
#     boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
#     """
#     # 1. Tile boxes2 and repeat boxes1. This allows us to compare
#     # every boxes1 against every boxes2 without loops.
#     # TF doesn't have an equivalent to np.repeat() so simulate it
#     # using tf.tile() and tf.reshape.
#     # [N, 1, 6]
#     b1s = tf.expand_dims(boxes1, 1)
#     # [1, M, 6]
#     b2s = tf.expand_dims(boxes2, 0)
#
#
#     # 2. Compute intersections
#     b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = np.split(b1, 6, axis=1)
#     b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = np.split(b2, 6, axis=1)
#     z1 = np.maximum(b1_z1, b2_z1)
#     y1 = np.maximum(b1_y1, b2_y1)
#     x1 = np.maximum(b1_x1, b2_x1)
#     z2 = np.minimum(b1_z2, b2_z2)
#     y2 = np.minimum(b1_y2, b2_y2)
#     x2 = np.minimum(b1_x2, b2_x2)
#     intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
#     # 3. Compute unions
#     b1_area = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
#     b2_area = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
#     union = b1_area + b2_area - intersection
#     # 4. Compute IoU and reshape to [boxes1, boxes2]
#     iou = intersection / union
#     overlaps = np.reshape(iou, [np.shape(boxes1)[0], np.shape(boxes2)[0]])
#     return overlaps



#
#
# def compute_overlaps(boxes1, boxes2):
#     """Computes IoU overlaps between two sets of boxes.
#     boxes1, boxes2: [N, (y1, x1, y2, x2)].
#
#     For better performance, pass the largest set first and the smaller second.
#     """
#     # Areas of anchors and GT boxes
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
#
#     # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
#     # Each cell contains the IoU value.
#     overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
#     for i in range(overlaps.shape[1]):
#         box2 = boxes2[i]
#         overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
#     return overlaps
#
#
# def compute_iou(box, boxes, box_area, boxes_area):
#     """Calculates IoU of the given box with the array of the given boxes.
#     box: 1D vector [y1, x1, y2, x2]
#     boxes: [boxes_count, (y1, x1, y2, x2)]
#     box_area: float. the area of 'box'
#     boxes_area: array of length boxes_count.
#
#     Note: the areas are passed in rather than calculated here for
#     efficiency. Calculate once in the caller to avoid duplicate work.
#     """
#     # Calculate intersection areas
#     y1 = np.maximum(box[0], boxes[:, 0])
#     y2 = np.minimum(box[2], boxes[:, 2])
#     x1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[3], boxes[:, 3])
#     intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
#     union = box_area + boxes_area[:] - intersection[:]
#     iou = intersection / union
#     return iou



############################################################
#  Relation Module
############################################################



class ParserStatisticsData():
    def __init__(self, path):
        self.path = path

        lower_data = "lower_teeth.pkl"
        upper_data = "upper_teeth.pkl"

        with open(os.path.join(path, lower_data), "rb") as f:
            self.lower_data = pickle.load(f)

        with open(os.path.join(path, upper_data), "rb") as f:
            self.upper_data = pickle.load(f)

        assert isinstance(self.upper_data, TeethStructure)
        assert isinstance(self.lower_data, TeethStructure)


        self.teeth_box = {}
        self.teeth_voxel = {}
        # TODO: upper & lower alingment offset value ( z-axis)
        self.lower_offset = 150
        # TODO
        self.spacing = 0.2
        self.shift = np.array([self.lower_offset, 0, 0,
                               self.lower_offset, 0, 0 ])

        lowerdata_shape = np.array(self.lower_data.voxel.shape) + self.shift[:3]
        upperdata_shape = np.array(self.upper_data.voxel.shape)
        self.boundary = np.maximum(lowerdata_shape, upperdata_shape)

        self._prepare()

    def get_shape(self):
        """
        the boundary including upper voxel, lower voxel
        lower-voxel is shifted by offset-value
        :return:
        """

        return self.boundary


    def _prepare(self):
        upper_tau = []
        lower_tau = []
        for i in range(1, 5):
            for j in range(1, 9):
                if i < 3:
                    upper_tau.append( i * 10 + j)
                else:
                    lower_tau.append(i * 10 + j)

        self._extract_data(upper_tau, self.upper_data.voxel)
        self._extract_data(lower_tau, self.lower_data.voxel)

    def _extract_data(self, taus, data):
        offset = np.array([-3, -3, -3, 3, 3, 3])
        shape_max = np.array(data.shape) - np.array([1, 1, 1])
        shape_max = np.concatenate([shape_max, shape_max])

        for tx in taus:
            ixs = np.where(data == tx)
            ixs = np.stack(ixs, axis=-1)
            if ixs.size > 0:
                min_p, max_p = np.min(ixs, axis=0), np.max(ixs, axis=0)
                box = np.concatenate([min_p, max_p]) + offset
                box = np.clip(box, np.zeros_like(shape_max), shape_max )
                z1, y1, x1, z2, y2, x2 = box
                self.teeth_box[tx] = (box)
                self.teeth_voxel[tx] = data[z1:z2, y1:y2, x1:x2] == tx


    def read_teeth(self, tau):
        assert tau in self.teeth_voxel
        shape = self.upper_data.voxel.shape if tau < 30 else self.lower_data.voxel.shape
        shift = np.zeros_like(self.shift) if tau < 30 else self.shift
        box = self.teeth_box[tau] + shift
        return self.teeth_voxel[tau], box, shape



class DataGenrator(object):
    def __init__(self, max_rois, pool_shape=[24, 24, 24]):
        datapath = "./data"
        # curpath = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), datapath)
        try:
            self.statistics_data = ParserStatisticsData(target_path)
        except Exception as e:
            logger = get_runtime_logger()
            logger.error(e)
        self.num_rois = max_rois

        sample_data = os.path.join(target_path, "../data/relationdata.pkl")
        if os.path.exists(sample_data):
            with open(sample_data, "rb") as f:
                self.relation_samples = pickle.load(f)
        # pool_shape = [24, 24, 24]
        cropped_voxels, boxes, shape, taus = self.get_statistics_items()
        rsz_voxel = [image_utils.get_interpolation(vox, pool_shape) for vox in cropped_voxels]
        rsz_voxel = np.array(rsz_voxel)
        rsz_voxel = np.expand_dims(rsz_voxel, axis=-1)
        self.moving_teeth = rsz_voxel
        self.moving_taus = taus
        self.batch_size = 80

        np.random.seed(int(time.time()))

    def get_datasize(self):
        return len(self.relation_samples)

    def _next_batch(self):
        """

        :return:
        roi-pooled feature for mask [N, D, H, w]
        roi-pooled source [N, D, H, W]
        ground truth mask [N, D, H, w]
        roi-pool feature-map for class [N, d, h, w]
        rois bounding box [N, 6]
        target ground truth label [N]

        """

        i = np.random.choice(len(self.relation_samples))
        rdata = self.relation_samples[i]
        feats = [rdata["feat_mask"],
                rdata["source"],
                rdata["gt_mask"],
                rdata["feat_class"],
                rdata["rois"],
                rdata["ids"],
                ]
        return feats

    def next_source(self):
        while True:
            i = np.random.choice(len(self.relation_samples))
            rdata = self.relation_samples[i]
            gt_masks = rdata["gt_mask"]
            src = rdata["source"]
            label = rdata["ids"]
            masks = rdata["feat_mask"]
            real_src = rdata["source"]
            k = np.random.choice(src.shape[0])
            source = src[k]
            mask = masks[k]
            tau = label[k]
            gt_mask = gt_masks[k]
            # print(i, k)
            real_src = real_src[k]
            if not tau in self.moving_taus:
                continue
            target = self.moving_teeth[self.moving_taus.index(tau)]
            source = source.reshape([1, *source.shape, 1])
            target = target.reshape([1, *target.shape[:3], 1])
            mask = mask.reshape([1, *mask.shape[:3], 1])
            real_src = real_src.reshape([1, *real_src.shape[:3], 1])
            gt_mask = gt_mask.reshape([1, *gt_mask.shape, 1])
            return source, target, mask, real_src, gt_mask

    def __next__(self):
        items = self._next_batch()
        feats = items[3]
        rois = items[4]
        class_ids =items[5]

        # generate fixed out rois
        if feats.shape[0] < self.num_rois:
            p = self.num_rois - feats.shape[0]
            feats = np.pad(feats, [[0, p], [0, 0], [0, 0], [0, 0], [0, 0]])
            rois = np.pad(rois, [[0, p], [0, 0]])
            class_ids = np.pad(class_ids, [[0, p]])
        elif feats.shape[0] > self.num_rois:
            p = self.num_rois
            feats = feats[:p]
            rois = rois[:p]
            class_ids = class_ids[:p]
        else:
            pass

        feats = np.expand_dims(feats, axis=0)
        rois = np.expand_dims(rois, axis=0)
        class_ids = np.expand_dims(class_ids, axis=0)

        fdi2label = get_fdi2label()
        class_label = fdi2label[class_ids]
        return [feats, rois, class_label]

    def next(self):
        return self.__next__()

    def get_statistics_items(self):
        """

        :return:
        the list of voxel , each voxel is different size [N, voxels]
        array of boxes, [N, 6],
        N = 28 = 7 * 4
        """
        taus = []
        # except windowm-teeth
        for i in range(1, 5):
            for j in range(1, 8):
                taus.append(i*10+j)

        voxels = []
        boxes = []

        for tau in taus:
            voxel, box, shape  = self.statistics_data.read_teeth(tau)
            voxels.append(voxel)
            boxes.append(box)
        shape = self.statistics_data.get_shape()
        boxes = np.array(boxes)
        boxes = norm_boxes(boxes, shape)
        # voxel = np.array(voxel)
        return voxels, boxes, shape, taus

def get_statistics_feat(config):
    """
    :param config:
    :return:
    num = 28 teeth
    feature [num, d, h, w, 1]
    rois [num, 6]
    """

    max_rois = 80
    datagen = DataGenrator(max_rois)
    cropped_voxels, boxes, shape, taus = datagen.get_statistics_items()
    rsz_voxel = [image_utils.get_interpolation(vox, config.POOL_SHAPE) for vox in cropped_voxels]
    rsz_voxel = np.array(rsz_voxel)
    rsz_voxel = np.expand_dims(rsz_voxel, axis=-1)

    return rsz_voxel, boxes
