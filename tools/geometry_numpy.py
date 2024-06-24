import numpy as np
from typing import List, Tuple, Union
from scipy.ndimage import sobel
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation

import time
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import vtk
from vtkmodules.util import numpy_support
from typing import List

from tools import InterpolateWrapper
# from tools.trilinear.InterpolateWrapper import InterpolateWrapper

from . import vtk_utils
from . import utils_numpy as utils
# from trainer import utils_numpy as utils
# from trainer import vtk_utils
# from . import image_utils

from commons import timefn2, get_runtime_logger

def decompose_general_boxes(obb):
    """
    :param obb:[N, 9] oriented bounding box
    :return:
    """
    # tf.assert_equal(tf.shape(obb)[-1], 9)
    # print(obb.shape)
    return obb[..., :3], obb[..., 3:6], obb[..., 6:]

def translation_mat(translation):
    dim = translation.size
    mat = np.eye(dim + 1)
    mat[:dim, dim] = translation
    return mat


def scaling_mat(scale):
    scale = np.asarray(scale)
    dim = scale.size
    mat = np.eye(dim + 1)
    mat[:dim, :dim] = np.diag(scale)
    return mat


def rotation_mat(rot):
    dim = rot.shape[0]
    mat = np.eye(dim + 1)
    mat[:dim, :dim] = rot
    return mat


def decompose_matix2euler2(rotmat):
    """
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    https://www.gregslabaugh.net/publications/euler.pdf   ,
    Computing Euler angles from a rotation matrix Gregory G. Slabaugh
    :param rotmat: [N, 3, 3] or [3, 3]
    :return: [N, 3] or [3], (wz, wy, wx)
    """
    # rotmat = np.ones([10, 3, 3])
    if rotmat.ndim == 3:
        R = np.transpose(rotmat, [1, 2, 0])
    else:
        R = rotmat

    # singular = sy < 1e-6
    R31 = R[2, 0]
    sign_R31 = np.sign(R31)
    singular_check = np.logical_not(np.isclose(np.abs(R31), 1.0))
    wy = np.where(singular_check, -np.arcsin(R31), sign_R31 * np.pi/2)
    coswy = np.cos(wy)


    wz = np.where(singular_check,
                  np.arctan2(R[1, 0] / coswy, R[0, 0]/coswy),
                  0.)

    wx = np.where(singular_check,
                  np.arctan2(R[2, 1] / coswy, R[2, 2]/coswy),
                  wz * sign_R31 + np.arctan2(R[0, 1]*sign_R31, R[0, 2]*sign_R31))

    axis = rotmat.ndim - 2
    theta = np.stack([wz, wy, wx], axis=axis)
    return theta


class OrientedBoundingBox(object):
    def __init__(self, num):
        self.data = np.array([0, 9])
        self.normalized = False

    def create(self, centers, size, theta, normalized):
        assert centers.shape == size.shape == theta.shape and centers.ndim == 1
        self.normalized = normalized
        self.data = np.stack([centers, size, theta], axis=1)

    def array(self):
        return self.data


def create_vtk_obb_cube(bboxes):
    """
    :param bboxes:[8]
    :return:
    """
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    numpy_points = bboxes
    # 6 plane cell of bboxes
    numpy_cells = np.array([
        [4, 0, 1, 3, 2],
        [4, 0, 1, 5, 4],
        [4, 1, 3, 7, 5],
        [4, 0, 2, 6, 4],
        [4, 2, 3, 7, 6],
        [4, 4, 5, 7, 6],
    ])
    # z, y,x ------> x, y, z
    vtk_points_array = numpy_support.numpy_to_vtk(numpy_points[:, ::-1])
    vtk_cell_array = numpy_support.numpy_to_vtk(numpy_cells.reshape([-1, 5]), array_type=vtk.VTK_ID_TYPE)
    points.SetData(vtk_points_array)
    # 6 plane of boxes
    polys.SetCells(6, vtk_cell_array)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    obb_cube = vtk_utils.polydata2actor(polydata)
    obb_cube.GetMapper().ScalarVisibilityOff()

    obb_cube.GetProperty().SetRepresentationToWireframe()
    obb_cube.GetProperty().SetColor(tuple(np.random.uniform(0, 1, 3)))
    obb_cube.GetProperty().SetLineWidth(3)
    return obb_cube


def test_draw_patch(polys):
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    patches = []

    N = 5
    # for i in range(N):
    #
    for poly in polys:
        polygon = Polygon(poly, True)
        patches.append(polygon)

    colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.4)
    p.set_array(np.array(colors))

    fig, ax = plt.subplots()
    ax.add_collection(p)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    fig.colorbar(p, ax=ax)

    plt.show()


def rotation(dw):
    # eye = np.eye(2)
    # for i in range(2):
    cs = np.cos(dw)
    si = np.sin(dw)
    # si = np.sin(dw[i])
    rot = np.array([
        [cs, -si],
        [si, cs]
    ])
    t = np.eye(3)
    t[:2, :2] = rot
    return t


def translate(tx):
    tx = np.asarray(tx)
    t = np.eye(3)
    size = tx.size
    t[:size, -1] = tx
    return t


def scale(s):
    s = np.asarray(s)
    s = np.concatenate([s, [1]])
    return np.diag(s)


# sequence scaling-->rotation->translation
def transform(dw, tx, s):
    return np.matmul(translate(tx), np.matmul(rotation(dw), scale(s)))


def apply_transform(pts, t):
    tranfsorm_pts = np.matmul(t[:-1, :-1], pts.T).T + t[:-1, -1]
    return tranfsorm_pts


def transform_bbox(box1, dw, dt, ds):
    """
    :param box1: polygon rectangular bbox, NX d
    :param dw: orientation
    :param dt: traslaation
    :param ds: scale
    :return:
    """
    origin = box1.mean(axis=0)
    t1 = translate(-origin)
    dt2 = origin + dt
    t = transform(dw, dt2, ds)
    t = np.matmul(t, t1)

    transform_box = apply_transform(box1, t)
    return transform_box


def test_draw_box_and_rotate():
    box1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    box2 = transform_bbox(box1, np.pi / 10, [0.0, 0.0], [1.2, 2.0])
    # print(box1.shape)
    print(box1.shape, box2.shape)
    print(box2)
    test_draw_patch([box1, box2])


def create_sphere(grid, origin, radius):
    r = np.sqrt(np.sum((grid - origin) ** 2, axis=-1))
    return r < radius


def test_sphere_iou():
    x = np.linspace(0, 10, 150)
    y = np.linspace(0, 10, 150)
    z = np.linspace(0, 10, 150)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    meshs = np.stack([Z, Y, X], axis=-1)
    origin1 = np.array([3, 4, 5])
    radius1 = 2

    origin2 = np.array([7, 7, 8])
    radius2 = 1.5

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    elem_volume = dx * dy * dz

    sp1 = create_sphere(meshs, origin1, radius1)
    sp2 = create_sphere(meshs, origin2, radius2)

    # vtk_utils.visulaize(sp1*255, 123)

    # https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    print(sp1.shape, sp2.shape, sp1.dtype)
    # print( np.sum(sp1 == sp2))
    # print(np.sum(np.logical_and(sp1, sp2)))
    t1 = time.time()
    AB = np.sum(np.logical_and(sp1, sp2))
    t2 = time.time()
    A = np.sum(sp1)
    B = np.sum(sp2)
    print("----->", AB / (A + B), AB)

    d = np.linalg.norm(origin1 - origin2)
    R = radius1
    r = radius2

    compure_area = AB * elem_volume
    t3 = time.time()
    # R = 5
    # r = 3
    # d = 4

    area = np.pi * ((R + r - d) ** 2) * (d ** 2 + 2 * d * r - 3 * (r ** 2) + 2 * d * R + 6 * r * R - 3 * (R ** 2)) / (
                12 * d)

    if d > np.maximum(R, r):
        area = 0
    t4 = time.time()
    print(t2 - t1, t4 - t3)
    print("--->area", area, compure_area)

    s1 = vtk_utils.numpyvolume2vtkvolume(sp1 * 255, 123)
    s2 = vtk_utils.numpyvolume2vtkvolume(sp2 * 255, 123)
    vtk_utils.show_actors([s1, s2])


def compute_circle_iou(anchors, gt_center, gt_size):
    """
    :param anchors: [N,6]
    :param gt_center: [M, 3]
    :param gt_size: [M, 3]
    :return:
    """

    # origin = center - np.matmul(basis, (size - 1) / 2)
    # gt_center, size = split_boxes(gt_boxes)
    gt_min_size = np.min(gt_size, axis=-1)
    anchor_center = (anchors[:3] + anchors[3:]) / 2
    delta_center = np.expand_dims(anchor_center, axis=1) - np.expand_dims(gt_center, axis=0)
    distance_center = np.linalg.norm(delta_center, axis=-1)

    anchor_min_size = np.min(anchors[3:] - anchors[:3], axis=-1)
    ra = np.expand_dims(anchor_min_size, axis=1)
    rg = np.expand_dims(gt_size, axis=1)
    max_dist = np.minimum(ra, rg)
    RA, RG = np.meshgrid(ra, rg)

    D = distance_center
    area = np.pi * ((RA + RG - D) ** 2) * (
                D ** 2 + 2 * D * RG - 3 * (RG ** 2) + 2 * D * RA + 6 * RG * RA - 3 * (RG ** 2)) / (12 * D)
    iou = np.where(D < max_dist, area, 0)
    return iou


def rpn_targets(anchors, gt_class_ids, gt_boxes, config):
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_iou(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_iou(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    epsilon = 1e-6
    # TODO : except background? or not?
    rpn_match[anchor_iou_max < config.RPN_TRAIN_NEGAT_IOU] = -1
    # rpn_match[np.logical_and(0 < anchor_iou_max, anchor_iou_max < config.RPN_TRAIN_NEGAT_IOU)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= config.RPN_TRAIN_POSIT_IOU] = 1
    # print("positive iou", overlaps.max(axis=0))
    DEBUG = False

    if DEBUG:

        target_iou = np.max(overlaps, axis=0)
        counts = np.sum(overlaps > 0.5, axis=0)
        sort_ix = np.argsort(target_iou)
        for tau, iou, cnt, box in zip(gt_class_ids[sort_ix], target_iou[sort_ix], counts[sort_ix], gt_boxes[sort_ix]):
            box = box.astype(np.int)
            dhw = box[3:] - box[:3]
            print("{} : {} / #{} size : {}".format(tau, iou, cnt, dhw))
        print("--------------------")

    # np.argmax(anchor_iou_max, axis=0)

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    rpn_bbox[:ids.size] = utils.box_refinement(anchors[ids], gt_boxes[anchor_iou_argmax[ids]]) / \
                          config.RPN_BBOX_STD_DEV


# def test_random_x():

def normalize(x):
    x[:] = x / np.linalg.norm(x)
    return x


def generate_random_orthonormal():
    x = np.random.randn(3)
    normalize(x)
    ref = np.random.randn(3)
    normalize(ref)

    y = np.cross(x, ref)
    normalize(y)
    z = np.cross(x, y)
    base = np.stack([x, y, z], axis=0)
    return base


def generate_basis(vec:np.ndarray):
    assert vec.size == 3
    x = vec.copy()
    normalize(x)
    ref = np.random.randn(3)
    normalize(ref)

    y = np.cross(x, ref)
    normalize(y)
    z = np.cross(x, y)
    base = np.stack([x, y, z], axis=0)
    return base


def pivot_scale(scale:np.ndarray, center:np.ndarray):
    t0 = translation_mat(center)
    t2 = translation_mat(-center)
    t1 = scaling_mat(scale)
    return concat_transform([t0, t1, t2])


def decompose_matrix2euler_vtk(mat, return_radian=True):
    """
    compute euler angle from transform matrix,
    :param mat:[bz, by, bx] rotation matrix (orthonormal basis)
    :return: [3] [z, y, x] euler degree(radian)
    """

    if mat.shape == (3, 3):
        mat4x4 = np.eye(4)
        mat4x4[:3, :3] = mat
    elif mat.shape == (4, 4):
        mat4x4 = mat
    else:
        raise ValueError(mat.shape)
    t = vtk_utils.myTransform()
    t.set_from_numpy_mat(mat4x4)
    deg = np.array(t.GetOrientation()[::-1])
    if return_radian:
        deg = np.deg2rad(deg)

    return deg


def create_mat4x4_from_euler_vtk(theta):
    """
    :return:
    """
    deg = np.rad2deg(theta)
    tz = vtk_utils.myTransform()
    ty = vtk_utils.myTransform()
    tx = vtk_utils.myTransform()

    tz.RotateZ(deg[0])
    ty.RotateY(deg[1])
    tx.RotateX(deg[2])

    Tx, Ty, Tz = tx.convert_np_mat(), ty.convert_np_mat(), tz.convert_np_mat()

    mat4x4 = np.matmul(np.matmul(Tz, Tx), Ty)
    return mat4x4


def _____decompose_matix2euler(R):
    """
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    https://www.gregslabaugh.net/publications/euler.pdf   ,
    Computing Euler angles from a rotation matrix Gregory G. Slabaugh
    :param R:
    :return:
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:

        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([z, y, x])



def decompose_matix2euler2(rotmat):
    """
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    https://www.gregslabaugh.net/publications/euler.pdf   ,
    Computing Euler angles from a rotation matrix Gregory G. Slabaugh
    :param rotmat: [N, 3, 3] or [3, 3]
    :return: [N, 3] or [3], (wz, wy, wx)
    """
    # rotmat = np.ones([10, 3, 3])
    if rotmat.ndim == 3:
        R = np.transpose(rotmat, [1, 2, 0])
    else:
        R = rotmat

    # singular = sy < 1e-6
    R31 = R[2, 0]
    sign_R31 = np.sign(R31)
    singular_check = np.logical_not(np.isclose(np.abs(R31), 1.0))
    wy = np.where(singular_check, -np.arcsin(R31), sign_R31 * np.pi / 2)
    coswy = np.cos(wy)

    wz = np.where(singular_check,
                  np.arctan2(R[1, 0] / coswy, R[0, 0] / coswy),
                  0.)

    wx = np.where(singular_check,
                  np.arctan2(R[2, 1] / coswy, R[2, 2] / coswy),
                  wz * sign_R31 + np.arctan2(R[0, 1] * sign_R31, R[0, 2] * sign_R31))

    axis = rotmat.ndim - 2
    theta = np.stack([wz, wy, wx], axis=axis)
    return theta



def create_matrix_from_euler(theta):
    """
    :param theta: [N,3] or [3] array [N,(wz,wy,wx)]
    :return: [N,3,3] or [3,3] transform matrix
    """
    wz, wy, wx = np.split(theta, 3, axis=-1)
    # [N, 1]
    cosz = np.cos(wz)
    cosy = np.cos(wy)
    cosx = np.cos(wx)

    sinz = np.sin(wz)
    siny = np.sin(wy)
    sinx = np.sin(wx)

    zero = np.zeros_like(wz)
    one = np.ones_like(wz)

    # [3, 3, N, 1]
    rotz = np.array([
        [cosz, -sinz, zero],
        [sinz, cosz, zero],
        [zero, zero, one],
    ]
    )

    roty = np.array([
        [cosy, zero, siny],
        [zero, one, zero],
        [-siny, zero, cosy],
    ]
    )

    rotx = np.array([
        [one, zero, zero],
        [zero, cosx, -sinx],
        [zero, sinx, cosx]
    ]
    )

    # [3, 3, N, 1]-----> [3, 3, N]
    rotz, roty, rotx = [np.squeeze(p, axis=-1) for p in [rotz, roty, rotx]]
    if theta.ndim == 2:
        # [ N, 3, 3]
        rotz, roty, rotx = [np.transpose(p, [2, 0, 1]) for p in [rotz, roty, rotx]]

    rot = np.matmul(np.matmul(rotz, roty), rotx)
    if theta.ndim == 2:
        # affmat = np.repeat(np.expand_dims(rot, axis=0), rot.shape[0], 0)
        affmat = np.repeat(np.expand_dims(np.eye(4), axis=0), rot.shape[0], 0)
        affmat[:, :3, :3] = rot
    else:
        affmat = np.eye(4)
        affmat[:3, :3] = rot

    return affmat

def decompose_matrix(mat:np.ndarray)->List[np.ndarray]:
    """
    :param mat: 4x4 matrix
    :return: (rotation matrix (3, 3), scaling (3,), translation (3,)
    """
    assert mat.shape == (4, 4)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    scale = np.linalg.norm(rot, axis=-1)
    # BUG: more complex, incorrect. scaliing divison from rotation
    norm_rot = rot / np.linalg.norm(rot, axis=-1, keepdims=True)
    return norm_rot, scale, trans


def deompose_matrix(mat:np.ndarray)->List[np.ndarray]:
    """
    :param mat: 4x4 matrix
    :return: (rotation matrix (3, 3), scaling (3,), translation (3,)
    """
    logger = get_runtime_logger()
    logger.warn("this is typo name function. use 'decompose_matrix(...)'")
    return decompose_matrix(mat)


def compose_matrix(r, s, t):
    """

    :param r:
    :param s:
    :param t:
    :return:
    """
    sm = scaling_mat(s)
    rm = rotation_mat(r)
    tm = translation_mat(t)
    return concat_transform([tm, sm, rm])


def decompose_complete_matrix(mat):
    """
    T, R, S decompose matrix. d
    T (3,) R (3, 3) S (3, 3)
    using scipy.spatial.rotation
    :return:
    :rtype:
        rotation, scaling translation
    """
    assert mat.shape == (4, 4)

    rot_scale = mat[:3, :3]
    r = Rotation.from_matrix(rot_scale)

    only_rot = r.as_matrix()
    inv_rot = np.linalg.inv(only_rot)
    scale = np.dot(inv_rot, rot_scale)
    trans = mat[:3, 3]
    return only_rot, scale, trans


def compose_complete_matrix(rot, scale, trans):
    assert rot.shape == (3, 3)
    assert scale.shape == (3, 3)
    assert trans.shape == (3,)

    tmat = translation_mat(trans)
    rm = rotation_mat(rot)
    sm = rotation_mat(scale)
    return concat_transform([tmat, rm, sm])



def test_random_vec():

    np.random.seed(int(time.time()))
    for i in range(1000):
        base = generate_random_orthonormal()
        arg = np.argmax(np.abs(base), axis=1)

        mat = np.eye(4)
        mat[:3, :3] = base.T
        t = vtk_utils.myTransform()
        t.set_from_numpy_mat(mat)

        copyt = vtk_utils.myTransform()
        copyt.SetMatrix(t.GetMatrix())

        sad_decom = np.sum(np.abs(create_matrix_from_euler(decompose_matix2euler2(mat)) - mat))

        sad_decom2 = np.sum(np.abs(create_mat4x4_from_euler_vtk(decompose_matrix2euler_vtk(mat)) - mat))
        sad_decom3 = np.sum(np.abs(create_matrix_from_euler(decompose_matix2euler2(mat)) - mat))
        theta = np.random.uniform(-np.pi / 2, np.pi / 2, [3])
        sad_decom4 = np.sum(np.abs(decompose_matix2euler2(create_matrix_from_euler(theta)) - theta))
        # print("decompose difference-norm", np.isclose(sad_decom, 0), sad_decom)
        # print("decompose difference-vtk", np.isclose(sad_decom2, 0), sad_decom2)
        print("decompose difference-vtk", np.isclose(sad_decom3, 0), sad_decom3)
        print("decompose difference-vtk", np.isclose(sad_decom4, 0), sad_decom4)

        # print(decompose_matix(mat), decompose_matrix2euler(mat)[::-1])


def test_compare_iou():

    np.random.seed(int(time.time()))
    #
    for i in range(1000):
        base = generate_random_orthonormal()
        arg = np.argmax(np.abs(base), axis=1)

        # generate axis align bbox

        # genreate genreral bbox , affinte transform, shift and rotation

        # compute iou
        # 1. voxelization

        # 2. sphere approximation

        # 3. instance


def create_vertices_from_aabb(aabb)->np.ndarray:
    """
    :param aabb: (N, 6) aabb vertices (z1, y1, x1, z2, y2, x2)
    :return: similar with 'create_obb_pose_array(...)'.
    (N, 8, 3) with 8 vertices
    """

    # [N,1]
    z = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    x = np.linspace(-1, 1, 2)
    # y = np.linspace(0, dh, 2)
    # x = np.linspace(0, dw, 2)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    # [2, 2, 2, 3]
    zyx_grid = np.stack([Z, Y, X], axis=-1)

    bmin = aabb[:, :3]
    bmax = aabb[:, 3:]
    # (N, 3)
    fsize = bmax - bmin
    center = (bmin + bmax) / 2.

    # [1, 8, 3]
    zyx_pose = zyx_grid.reshape([1, -1, 3]) * np.expand_dims(fsize / 2, axis=1)
    # zyx_pose_transpose = np.transpose(zyx_pose, [0, 2, 1])

    # shift from center to origin
    bbox_vertices = np.expand_dims(center, axis=1) + zyx_pose
    return bbox_vertices



def transform_aabb(mat4x4:np.ndarray, aabb:np.ndarray, realign=True):
    """
    transform aabb matrix

    compute 8 vertices, and
    :param mat4x4:(4, 4)
    :param aabb:(N, 6)
    :return:
    """

    aabb_vertices = create_vertices_from_aabb(aabb)

    shape_aabb_verts =     aabb_vertices.shape
    reshaped_aabb_verts = aabb_vertices.reshape([-1, 3])

    trans_aabb_verts = apply_transform(reshaped_aabb_verts, mat4x4)
    # (N, 8, 3)
    trans_aabb_verts = trans_aabb_verts.reshape(shape_aabb_verts)

    if realign:

        bmin = np.min(trans_aabb_verts, axis=1)
        bmax = np.max(trans_aabb_verts, axis=1)
        return np.concatenate([bmin, bmax], axis=-1)
    else:
        raise NotImplementedError


def transform_obb(transform_mat, in_obb_meta):
    """
    rigid-transform OBb
    :param transform_mat:[4,4] rigid transforma matrix
    :param in_obb_meta: [N, 9] or [9] denormalized obb meta
    :return: transformed [N, 9] obb meta by given transformmat
    """

    ndim_obb = in_obb_meta.ndim
    if ndim_obb == 1:
        obb_meta = np.expand_dims(in_obb_meta, axis=0)
    else:
        obb_meta = in_obb_meta

    rot, scale, trans = decompose_matrix(transform_mat)
    sm = scaling_mat(scale)

    fctr, fsize, ftheta = decompose_general_boxes(obb_meta)
    fsize_scaled = apply_transform(fsize, sm)


    centers = obb_meta[:, :3]
    transform_centers = vtk_utils.apply_trasnform_np(centers, transform_mat)

    # t2 = np.array([vtk_utils.apply_trasnform_np(obb[i:i+1, :3], tm) for i, tm in enumerate(transform)])
    # t2 = np.squeeze(t2)

    # Compute basis of source-OBB, [N, 4, 4]
    bases = create_matrix_from_euler(obb_meta[:, 6:])

    rot, _, _ = decompose_matrix(transform_mat)
    rot_mat = rotation_mat(rot)
    # Affine transform basis of source-OBB
    basesT = np.transpose(bases, [0, 2, 1])
    mat_temp = np.matmul(rot_mat, basesT)
    mat = np.transpose(mat_temp, [0, 2, 1])

    # Decompose theta of orthonormal-basis
    transform_theta = decompose_matix2euler2(mat)
    # Same size, because rigid-transform
    meta = np.concatenate([transform_centers, fsize_scaled, transform_theta], axis=-1)

    # keep original ndim
    if ndim_obb == 1:
        meta = np.squeeze(meta, axis=0)

    return meta


def create_basis_transform(basis, origin):
    """
    :param basis: [3, 3]
    :param origin: [3]
    :return:
    """
    basisT = basis.T
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basisT
    transform_mat[:3, 3] = -np.dot(basisT, origin)
    return transform_mat


def create_trasnform(basis, origin):
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basis
    transform_mat[:3, 3] = -np.dot(basis, origin)
    return transform_mat


def _create_cube_from_obbmeta(obbmeta):
    """
    :param obbmeta:[9]
    :return:g
    """
    center, size, theta = decompose_general_boxes(obbmeta)
    # matrix = create_mat4x4_from_euler_vtk(theta)
    matrix = create_matrix_from_euler(theta)
    # matrix = create_matrix_from_euler(theta)

    basis = matrix[:3, :3]
    # basis z,y,x --->x,y,z and transpose
    return create_general_cube(center, size, basis[::-1, ::-1].T)


def create_cube_from_obbmeta(obbmeta):
    """
    :param obbmeta: denomalize_ obbmeta [N, 9] or [9]
    :return:
    """
    if obbmeta.ndim == 2:
        return [create_cube_from_obbmeta(o) for o in obbmeta]
    elif obbmeta.ndim == 1:
        return _create_cube_from_obbmeta(obbmeta)
    else:
        raise ValueError("dimension error", obbmeta.ndim)


def create_general_cube(center, size, basis):
    """
    :param center: center of bbox [3]
    :param size: bbox size [3]
    :param basis: [3,3] orthornoaml basis
    :return:
    """
    # get bounding bbox
    # fixed dhw
    # dhw = np.array([32, 15, 15])
    p1 = center - size / 2
    p2 = center + size / 2
    bbox = np.concatenate([p1, p2])
    vtk_bbox = vtk_utils.convert_box_norm2vtk(bbox[np.newaxis])
    cube = vtk_utils.get_cube(vtk_bbox[0])

    # shift from center to origin
    rot = create_trasnform(basis, np.zeros([3]))
    # rot = create_trasnform(pca.components_[::-1, ::-1].T, np.zeros([3]))
    rot_vtk = vtk_utils.myTransform()
    rot_vtk.set_from_numpy_mat(rot)

    vtk_origin = center[::-1]
    t1 = vtk_utils.myTransform()
    t1.Translate(*tuple(-vtk_origin))
    t2 = vtk_utils.myTransform()
    t2.Translate(*tuple(vtk_origin))

    concat_t = vtk_utils.myTransform()
    concat_t.Concatenate(t2)
    concat_t.Concatenate(rot_vtk)
    concat_t.Concatenate(t1)

    cube.SetUserTransform(concat_t)
    return cube


def create_obb_pose_from_meta(obbmeta):
    """
    :param obbmeta:[N,9]
    :return:[N, 8, 3] obb vertices
    """
    centers, fsize, theta = decompose_general_boxes(obbmeta)

    mat = create_matrix_from_euler(theta)

    if obbmeta.ndim == 2:
        # orthonormal (row major)--->  (col-major)
        basis = np.transpose(mat[:, :3, :3], [0, 2, 1])
    else:
        basis = mat.T

    return create_obb_pose_array(centers, fsize, basis)




def test_check_orthogonal_obb_pose(obb_poses):
    p0 = obb_poses[:, 0]
    p1 = obb_poses[:, 1]
    p2 = obb_poses[:, 2]
    p3 = obb_poses[:, 2]
    p4 = obb_poses[:, 4]
    p5 = obb_poses[:, 5]
    p6 = obb_poses[:, 6]
    p7 = obb_poses[:, 7]

    p10 = p1 - p0
    p32 = p3 - p2
    p54 = p3 - p2

    p51 = p3 - p2
    p73 = p7 - p3

    # (N, 3)
    x0 = p1 - p0
    y0 = p2 - p0
    z0 = p4 - p0

    print(np.dot(x0, y0.T))
    print(np.dot(y0, z0.T))
    print(np.dot(z0, x0.T))

    print(p10.shape)
    print(np.dot(p10, p32.T))
    print(np.dot(p10, p54.T))
    print(np.dot(y0, p51.T))
    print(np.dot(y0, p73.T))

def create_obb_from_obb_pose_array(obb_poses):
    """
    restored obb-meta from obb 8 vertidces
    compute 8 vertices from orientation bounding box.
    (x0, x1) // (y0, y1) // (z0, z1)
    p0 (z0, y0, x0)
    p1 (z0, y0, x1)
    p2 (z0, y1, x0)
    p3 (z0, y1, x1)
    p4 (z1, y0, x0)
    p5 (z1, y0, x1)
    p6 (z1, y1, x0)
    p7 (z1, y1, x1)
    :param obb_poses: (N, 8, 3) pose-array of obb-meta (ref.)  create_obb_pose_array_from_obbmeta(...) or create_obb_pose_array(...)
    :return: (N, 9)
    """
    # (N, 3)
    p0 = obb_poses[:, 0]
    p1 = obb_poses[:, 1]
    p2 = obb_poses[:, 2]
    p4 = obb_poses[:, 4]
    p7 = obb_poses[:, 7]

    # (N, 3)
    x0 = p1 - p0
    y0 = p2 - p0
    z0 = p4 - p0

    centers = (p0 + p7)/2.

    # (N, 1)
    x_len = np.linalg.norm(x0, axis=-1, keepdims=True)
    y_len = np.linalg.norm(y0, axis=-1, keepdims=True)
    z_len = np.linalg.norm(z0, axis=-1, keepdims=True)

    fsize = np.concatenate([z_len, y_len, x_len], axis=-1)

    x1 = x0 / x_len
    y1 = y0 / y_len
    z1 = z0 / z_len

    rot = np.stack([z1, y1, x1], axis=1)

    theta = decompose_matix2euler2(rot)

    return np.concatenate([centers, fsize, theta], axis=-1)

def create_trasnform4x4_from_obb(obbmeta):
    centers, fsize, theta = decompose_general_boxes(obbmeta)
    mat = create_matrix_from_euler(theta)
    matT = np.transpose(mat, [0, 2, 1])
    matT[:, :3, 3] = centers
    return matT


def affine_transform(pts, mat):
    """
    :param pts:[N, M, D]
    :param mat: [N, D+1, D+1]
    :return:
    """

    assert pts.ndim == 3 and mat.ndim == 3
    assert pts.shape[0] == mat.shape[0]
    d = pts.shape[-1]

    # [N, D
    ptsT = np.transpose(pts, [0, 2, 1])
    vtx = np.transpose(np.matmul(mat[:, :d, :d], ptsT), [0, 2, 1]) + np.expand_dims(mat[:, :d, d], axis=1)
    return vtx


def create_obb_pose_array(center, fsize, basis):
    """
    object oriented bounding box
    :param center: [N, 3] the centeor fo boxes
    :param fsize: [N, 3] the size of boxes
    :param basis: [N, 3, 3] the orthonormal basis of boxes
    :return: vertices of obb, [N, 8, 3]
    """
    _1dim =  center.ndim == 1 and fsize.ndim == 1 and basis.ndim == 2
    if _1dim:
        center = center[np.newaxis]
        fsize = fsize[np.newaxis]
        basis = basis[np.newaxis]

    # [N,1]
    z = np.linspace(-1, 1, 2)
    y = np.linspace(-1, 1, 2)
    x = np.linspace(-1, 1, 2)
    # y = np.linspace(0, dh, 2)
    # x = np.linspace(0, dw, 2)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    # [2, 2, 2, 3]
    zyx_grid = np.stack([Z, Y, X], axis=-1)

    # [1, 8, 3]
    zyx_pose = zyx_grid.reshape([1, -1, 3]) * np.expand_dims(fsize / 2, axis=1)
    zyx_pose_transpose = np.transpose(zyx_pose, [0, 2, 1])

    # shift from center to origin
    bbox_vertices = np.expand_dims(center, axis=1) + \
                    np.transpose(np.matmul(basis, zyx_pose_transpose), [0, 2, 1])

    if _1dim:
        bbox_vertices = np.squeeze(bbox_vertices)
    return bbox_vertices


def create_bbox_pose(center, fsize, basis):
    # center = np.empty([10, 3])
    # fsize = np.empty([10, 3])
    # basis = np.empty([10, 3, 3])
    dd, dh, dw = fsize
    z = np.linspace(0, dd, 2)
    y = np.linspace(0, dh, 2)
    x = np.linspace(0, dw, 2)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # shift from center to origin
    origin = center - np.matmul(basis, fsize / 2)
    grid = np.stack([Z, Y, X], axis=-1)
    grid_reshape = grid.reshape([-1, 3])
    pose = np.matmul(basis, grid_reshape.T).T + origin
    return pose


def norm_obb(obbmeta, shape):
    """
    지금도 헷갈리네.. normalize obb-meta
    [0, 1, 2, 4, 5]
    in order to include entire-region of image
    center - (2,)
    fsize - (4,) ->
    bounding box
    (2 - 4/2, 2 + 4/2)
    5 / (4 + 1)

    image_shape (5, )
    pose_scae = (5 - 1,)
    :param obbmeta:
    :param shape:
    :return:
    """
    centers, size, theta = decompose_general_boxes(obbmeta)
    size_scale = np.asarray(shape)
    pose_scale = size_scale - 1.0

    norm_centers = centers / pose_scale
    norm_size = size / pose_scale
    meta = np.concatenate([norm_centers, norm_size, theta], axis=-1)
    return meta


def denorm_obb(obbmeta, shape):
    centers, size, theta = decompose_general_boxes(obbmeta)
    size_scale = np.asarray(shape)
    pose_scale = size_scale - 1.0

    norm_centers = centers * pose_scale
    norm_size = size * pose_scale
    meta = np.concatenate([norm_centers, norm_size, theta], axis=-1)
    return meta

def aabb2obb(aabb_boxes):
    """
    :param aabb_boxes:
    :type aabb_boxes: (N, 6) or (6) boxees
    :return:
    :rtype:
    """
    _1dim = aabb_boxes.ndim == 1
    if _1dim:
        aabb_boxes = np.expand_dims(aabb_boxes, axis=0)

    fsize = aabb_boxes[:, 3:] - aabb_boxes[:, :3]
    center = ( aabb_boxes[:, 3:] + aabb_boxes[:, :3] ) / 2.
    obb = np.concatenate([center, fsize, np.zeros_like(fsize)], axis=-1)
    if _1dim:
        obb = np.squeeze(obb)
    return obb

def obb2aabb(obb_boxes):
    """
    convert obb(orientation bounding box) to aabb (axis-aligned bounding box)
    :param obb_boxes: (9,) or (num_rois, 9)
    :return: aabb // (6,) or (num_rois, 6)
    """
    if obb_boxes.ndim == 1:
        ex_obb_boxes = np.expand_dims(obb_boxes, axis=0)
    else:
        ex_obb_boxes = obb_boxes


    obb_pose_res = create_obb_pose_from_meta(ex_obb_boxes)
    bmin = np.min(obb_pose_res, axis=1)
    bmax = np.max(obb_pose_res, axis=1)
    aabb = np.concatenate([bmin, bmax], axis=1)
    if obb_boxes.ndim == 1:
        aabb = np.squeeze(aabb, axis=0)
    return aabb

# def align_warps(obb, meta_shape, metho):
#
#     obb = denorm_obb(obb, meta_shape)
#     shape = np.asarray(pool_shape)
#     center, fsize, theta = decompose_general_boxes(obb)
#     # matrix = create_mat4x4_from_euler_vtk(theta)
#     matrix = create_matrix_from_euler(theta)
#
#     # orthonormal basis : row-vector ----> colume-vector
#     basis = np.transpose(matrix[:, :3, :3], [0, 2, 1])
#
#     mz, my, mx = shape[0], shape[1], shape[2]
#     # [N,1]
#     z = np.linspace(-1., 1., mz)
#     y = np.linspace(-1., 1., my)
#     x = np.linspace(-1., 1., mx)
#     # y = np.linspace(0, dh, 2)
#     # x = np.linspace(0, dw, 2)
#
#     Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
#     # [d, h, w, 3]
#     zyx_grid = np.stack([Z, Y, X], axis=-1)
#
#     # [N, d*h*w, 3]
#     # scalding size
#     size_scale = (fsize - 1)/2.
#     zyx_pose = np.reshape(zyx_grid, [1, -1, 3]) * np.expand_dims(size_scale, axis=1)
#
#     basisT = np.transpose(basis,  [0, 2, 1])
#     # shift from center to origin1
#     # [N, d*h*w, 3]
#     warp_points = np.expand_dims(center, axis=1) + np.matmul(zyx_pose, basisT)
#     # [N*d*h*w, 3]
#     reshape_warp_points = np.reshape(warp_points, [-1, 3])

def voi_align_from_aabb(volume, normed_aabb, meta_shape, pool_shape, method='linear',
                        interpolate_func: RegularGridInterpolator = None, return_affine_transform=False):
    """
    voi_align_from_obb(...) 이용해서 resampling
    converting form aabb to obb. then apply function 'voi_align_from_obb'
    :param volume: (D,H,W) volume
    :type volume: np.ndarray
    :param normed_aabb: (N, 4) aabb
    :type normed_aabb: np.ndarray
    :param meta_shape:
    :type meta_shape:
    :param pool_shape:
    :type pool_shape:
    :param method:
    :type method:
    :param interpolate_func:
    :type interpolate_func:
    :param return_affine_transform:
    :type return_affine_transform:
    :return:
    :rtype:
    """
    raise ValueError('moved image_utils.py::voi_align_from_aabb')
    # normed_obb = aabb2obb(normed_aabb)

def voi_align_from_obb(volume, obb, meta_shape, pool_shape, method="linear",
                       interpolate_func: InterpolateWrapper = None, return_affine_transform=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    resize ratio = (pool_shape - 1) / fsize
    fsize is size of bounding-box
    exmaple) consider 100 X 100 2d image, and obb (49.5, 49.5, 99, 99), crop and resize as (100, 100)
    then  (100 -1) / 99 = 1.0 . then resize raatio 1.0

    :param volume: [D, H, W]
    :param obb: [num_rois, 9]
    :param meta_shape: used to denomalize obb-meta
    :param pool_shape: mask shape interger [d, h, w]
    :return: [num_rois, d, h, w]
    """
    raise ValueError('moved image_utils.py::voi_align_from_obb')



def compute_obb_regressor_loss_overlaps(obbmeta1, obbmeta2, name=""):
    """
    :param obbmeta1: [N, 9]
    :param obbmeta2: [M, 9]
    :param name:
    :return: [N, M]
    """
    # with tf.name_scope(name):
    split_meta1 = decompose_general_boxes(obbmeta1)
    split_meta2 = decompose_general_boxes(obbmeta2)

    # [N, 1, 3]
    center1, size1, theta1 = [np.expand_dims(x, axis=1) for x in split_meta1]
    # [1, M, 3]
    center2, size2, theta2 = [np.expand_dims(x, axis=0) for x in split_meta2]

    # [N, M, 3]
    dzyx = (center2 - center1) / size1
    ddhw = np.log(size2 / size1)
    dw = theta2 - theta1

    def smooth_l1_array(x):
        absx = np.abs(x)
        return np.where(np.less(absx, 1.), absx, np.power(x, 2))

    # [N, M, 9]
    deltas = np.concatenate([dzyx, ddhw, dw], axis=-1)
    return np.exp(-np.sum(smooth_l1_array(deltas), axis=-1))


def sampler(volume, center, size, basis, method="linear"):
    assert method in ["linear", "nearest"]
    dd, dh, dw = size
    z = np.linspace(0, dd, dd - 1)
    y = np.linspace(0, dh, dh - 1)
    x = np.linspace(0, dw, dw - 1)

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    origin = center - np.matmul(basis, (size - 1) / 2)
    grid = np.stack([Z, Y, X], axis=-1)
    grid_reshape = grid.reshape([-1, 3])
    pose = np.matmul(basis, grid_reshape.T).T + origin

    vol = np.squeeze(volume)
    d, h, w = vol.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    source_interpolation = RegularGridInterpolator((mz, my, mx), vol, method=method, bounds_error=False, fill_value=0)
    resample_array = source_interpolation(pose)
    resample_array = resample_array.reshape(grid.shape[:3])

    return resample_array



def compute_intersection_from_meta(obbmeta1, obbmeta2):
    center1, size1, theta1 = decompose_general_boxes(obbmeta1)
    center2, size2, theta2 = decompose_general_boxes(obbmeta2)
    return compute_intersection(center1, size1, theta1, center2, size2, theta2)


def compute_intersection(center1, size1, theta1, center2, size2, theta2):
    """
    :param center1:
    :param size1:
    :param theta1:
    :param center2:
    :param size2:
    :param theta2:
    :return:
    """

    # [N, 4, 4]
    mat1 = create_matrix_from_euler(theta1)
    mat2 = create_matrix_from_euler(theta2)

    # orthonormal (row major)--->  (col-major)
    basis1 = np.transpose(mat1[:, :3, :3], [0, 2, 1])
    basis2 = np.transpose(mat2[:, :3, :3], [0, 2, 1])

    # [N, 8, 3]
    pose1 = create_obb_pose_array(center1, size1, basis1)
    # [M, 8, 3]
    pose2 = create_obb_pose_array(center2, size2, basis2)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # pts = pose2.reshape([-1, 3])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot(pts.T[0], pts.T[1], pts.T[2], 'go')

    distance = np.linalg.norm(np.expand_dims(pose1, axis=1) - np.expand_dims(pose2, axis=0),
                              axis=-1)
    # [N]
    R = np.min(size1, axis=1, keepdims=True)
    # [M]
    r = np.transpose(np.min(size2, axis=1, keepdims=True))
    # d = distance
    Mr = np.maximum(R, r)
    mr = np.minimum(R, r)
    sig = (Mr + mr) / 2 * 1.2
    mean_distance = np.mean(distance, axis=-1)

    iou = np.exp(-((mean_distance / sig) ** 2))

    return iou


def detection_targeting(obb1, gt_obb2):
    """
    :param obb1: [N, 9]
    :param gt_obb2: [M, 9]
    :return:
    """

    center1, size1, theta1 = decompose_general_boxes(obb1)
    center2, size2, theta2 = decompose_general_boxes(gt_obb2)
    # [M, N]
    iou = compute_intersection(center1, size1, theta1, center2, size2, theta2)
    return iou



def compute_sphere_intersection(anchor_meta1, gt_meta2):
    center1, size1, theta1 = np.split(anchor_meta1, 3, axis=-1)
    center2, size2, theta2 = np.split(gt_meta2, 3, axis=-1)
    min_size1 = np.min(size1, axis=1) / 2
    min_size2 = np.min(size2, axis=1) / 2

    # print(center1.shape, center2.shape)
    distance = cdist(center1, center2)
    m, n = distance.shape
    #
    # gt_size = gt_meta2.shape[0]
    # near_args = distance.argmax(axis=0)

    # print(distance.shape, m, n)
    # # nearest_centers = distance[near_args, np.arange(gt_size)]
    # # nearest_size = min_size1[near_args]
    # print(min_size2.shape, min_size1.shape)
    R = min_size1.reshape([m, 1, 1])
    r = min_size2.reshape([1, n, 1])
    neigh_dist = np.expand_dims(distance, axis=-1)
    d = neigh_dist

    from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # pts1 = center1.T
    # ax.plot(pts1[0][::50],pts1[1][::50], pts1[2][::50], 'r*')
    # pts2 = center2.T
    # ax.plot(pts2[0],pts2[1], pts2[2], 'go')

    area = np.pi * ((R + r - d) ** 2) * (
            d ** 2 + 2 * d * r - 3 * (r ** 2) + 2 * d * R + 6 * r * R - 3 * (R ** 2)) / (12 * d)

    A = np.pi * (R ** 3) * (4 / 3)
    B = np.pi * (r ** 3) * (4 / 3)

    subset = d < np.minimum(R, r)
    non_inter = d > np.maximum(R, r)

    A_B = A + B
    iou = area / A_B
    min_area = np.minimum(A, B)
    iou[subset] = min_area[subset] / (A_B[subset])
    iou[non_inter] = 0
    iou = np.squeeze(iou)

    return iou


def transform_general_bbox(meta, gt_meta):
    ctr, shape, theta = np.split(meta, 3, -1)
    gt_ctr, gt_shape, gt_theta = np.split(gt_meta, 3, -1)

    # center_z, center_y, center_x = np.split(ctr, 3, -1)
    # depth, height, width = np.split(shape, 3, -1)
    # theta_z, theta_y, theta_x = np.split(theta, 3, -1)
    #
    # gt_center_z, gt_center_y, gt_center_x = np.split(gt_ctr, 3, -1)
    # gt_depth, gt_height, gt_width = np.split(gt_shape, 3, -1)
    # gt_theta_z, gt_theta_y, gt_theta_x = np.split(gt_theta, 3, -1)

    dzyx = (gt_ctr - ctr) / shape
    ddhw = np.log(gt_shape / shape)
    dw = gt_theta - theta

    return np.concatenate([dzyx, ddhw, dw], axis=-1)



def invTransform_general_bbox(meta, deltas):
    ctr, shape, theta = np.split(meta, 3, -1)
    delta_ctr, delta_shape, delta_theta = np.split(deltas, 3, -1)

    out_ctr = ctr + delta_ctr * shape
    out_shape = np.exp(delta_shape) * shape
    out_theta = theta + delta_theta
    return np.concatenate([out_ctr, out_shape, out_theta], axis=-1)




def get_axes(scale=50, ctr=None):
    axes = vtk.vtkAxesActor()
    ctr = np.array([0., 0., 0.]) if ctr is None else ctr

    t1 = vtk.vtkTransform()
    t1.Scale(scale, scale, scale)

    t2 = vtk.vtkTransform()
    t2.Translate(tuple(ctr))

    t = vtk.vtkTransform()
    t.Concatenate(t2)
    t.Concatenate(t1)

    axes.SetUserTransform(t)
    return axes


def convert_obb2vtkcube(obb_meta):
    """
    :param obb_meta:[N,9]
    :return:
    """
    assert np.shape(obb_meta)[-1] == 9 and np.ndim(obb_meta) == 2
    center, size, theta = decompose_general_boxes(obb_meta)
    # matrix = create_mat4x4_from_euler_vtk(theta)
    matrix = create_matrix_from_euler(theta)

    # orthonormal (row major)--->  (col-major)
    basis = np.transpose(matrix[:, :3, :3], [0, 2, 1])

    general_bboxes = create_obb_pose_array(center, size, basis)

    obb_vtk_cubes = [create_vtk_obb_cube(b) for b in general_bboxes]
    return obb_vtk_cubes


def split_meta(obb_meta):
    center, fsize, theta = decompose_general_boxes(obb_meta)
    matrix = create_matrix_from_euler(theta)

    if obb_meta.ndim == 2:
        # row-->colume vector
        # basis = general_box.matrix[:3, :3].T
        basis = np.transpose(matrix[:, :3, :3], [0, 2, 1])
    elif obb_meta.ndim == 1:
        basis = matrix[:3, :3].T

    return basis, center, fsize, theta


def concat_transform(transform_list):
    """
    :param transform_list: list of array [4,4]
    :return:
    """
    mat = np.eye(4)
    for t in reversed(transform_list):
        mat = np.dot(t, mat)
    return mat


def reverse_transform(transform):
    t_rev = transform.copy()
    t_rev[:3, :3] = transform[2::-1, 2::-1]
    t_rev[:3, 3] = transform[2::-1, 3]
    return t_rev


def reverse_rotation(transform):
    ndim = transform.shape[0]
    t_rev = transform.copy()
    t_rev[:ndim, :ndim] = transform[(ndim-1)::-1, (ndim-1)::-1]
    return t_rev


@timefn2
def refinement_mask_pose_only(resample_volume, obb_meta, source_shape, method="nearest", labels=None,
                              return_normal=False, offset=1):
    """
    :param resample_volume: [N, d, h, w]
    :param obb_meta: origin of orthonormal basis [N, 3]
    :param source_shape: to refinement shape of volume, tuple of [3]
    :param offset: in order to include boundary of bounding box, enlarge sampling region a little
    :return: [source_shape],[D,H,W] restored full-mask
    """

    # pool_shape = np.array([v.shape for v in resample_volume])
    bases, centers, fsizes, theta = split_meta(obb_meta)

    # [num_rois, 8], compute extent of obb in aabb-domain
    obb_vertices = create_obb_pose_array(centers, fsizes, bases)

    max_inds = np.asarray(source_shape) - 1
    # offset = 1

    labels = np.ones([obb_meta.shape[0]], dtype=np.int) if labels is None else labels

    pose_dict = list()
    for i, (pts, pool_volume, origin, basis, fsize) in enumerate(
            zip(obb_vertices, resample_volume, centers, bases, fsizes)):
        # bbox =  np.concatenate(np.min(pts, axis=0), np.max(pts, axis=0), axis=0)
        p1 = np.floor(np.clip(np.min(pts, axis=0) - offset, 0, max_inds)).astype(np.int)
        p2 = np.ceil(np.clip(np.max(pts, axis=0) + offset, 0, max_inds)).astype(np.int)
        z1, y1, x1 = p1
        z2, y2, x2 = p2
        isize = p2 - p1 + 1
        dz, dy, dx = isize

        resample_scale = (np.asarray(pool_volume.shape) - 1) / fsize

        wz = np.linspace(z1, z2, dz)
        wy = np.linspace(y1, y2, dy)
        wx = np.linspace(x1, x2, dx)

        warps = np.stack(np.meshgrid(wz, wy, wx, indexing='ij'), axis=-1).reshape([-1, 3])

        sm = np.diag(np.concatenate([resample_scale, [1]]))

        # aligned_pts = apply_transform(warps, afm1)
        Tmat = create_basis_transform(basis, origin)

        # biases = (np.array(pool_volume.shape) - 1)/2.
        biases = fsize / 2
        bt = translation_mat(biases)

        # afm = np.matmul(sm, np.matmul(bt, Tmat))
        afm = concat_transform([sm, bt, Tmat])

        pose_afm = concat_transform([afm, translation_mat(p1)])
        inv_pose_afm = np.linalg.inv(pose_afm)
        concat_afms = concat_transform([translation_mat(p1), inv_pose_afm])
        # pose_afm_world = concat_transform([translation_mat(p1), inv_pose_afm])
        pose = np.argwhere(pool_volume > 0.5)
        # pose = np.argwhere(pool_volume > 0.5)
        transform_pose = apply_transform(pose, concat_afms)

        pose_dict.append(transform_pose)
        # pose_dict[labels[i]] = transform_pose

    return pose_dict


def refinement_mask_pose(resample_volume, obb_meta, source_shape, method="nearest", labels=None, return_normal=False):
    """
    :param resample_volume: [N, d, h, w]
    :param obb_meta: origin of orthonormal basis [N, 3]
    :param source_shape: to refinement shape of volume, tuple of [3]
    :return: [source_shape],[D,H,W] restored full-mask
    """

    # pool_shape = np.array([v.shape for v in resample_volume])
    bases, centers, fsizes, theta = split_meta(obb_meta)

    # [num_rois, 8, 3], compute extent of obb in aabb-domain
    obb_vertices = create_obb_pose_array(centers, fsizes, bases)

    max_inds = np.asarray(source_shape) - 1
    offset = 1

    full_refinement_mask = np.zeros(source_shape, dtype=np.uint8)

    labels = np.ones([obb_meta.shape[0]], dtype=np.int) if labels is None else labels

    DEBUG = False

    def _create_translate(translation):
        dim = translation.size
        mat = np.eye(dim + 1)
        mat[:dim, dim] = translation
        return mat

    pose_dict = {}
    normal_dict = {}
    for i, (pts, pool_volume, origin, basis, fsize) in enumerate(
            zip(obb_vertices, resample_volume, centers, bases, fsizes)):
        # bbox =  np.concatenate(np.min(pts, axis=0), np.max(pts, axis=0), axis=0)
        p1 = np.floor(np.clip(np.min(pts, axis=0) - offset, 0, max_inds)).astype(np.int)
        p2 = np.ceil(np.clip(np.max(pts, axis=0) + offset, 0, max_inds)).astype(np.int)
        z1, y1, x1 = p1
        z2, y2, x2 = p2
        isize = p2 - p1 + 1
        dz, dy, dx = isize

        resample_scale = (np.asarray(pool_volume.shape) - 1) / fsize

        wz = np.linspace(z1, z2, dz)
        wy = np.linspace(y1, y2, dy)
        wx = np.linspace(x1, x2, dx)

        warps = np.stack(np.meshgrid(wz, wy, wx, indexing='ij'), axis=-1).reshape([-1, 3])

        sm = np.diag(np.concatenate([resample_scale, [1]]))

        # aligned_pts = apply_transform(warps, afm1)
        Tmat = create_basis_transform(basis, origin)

        # biases = (np.array(pool_volume.shape) - 1)/2.
        biases = fsize / 2
        bt = _create_translate(biases)

        # afm = np.matmul(sm, np.matmul(bt, Tmat))
        afm = concat_transform([sm, bt, Tmat])

        pose_afm = concat_transform([afm, _create_translate(p1)])
        inv_pose_afm = np.linalg.inv(pose_afm)
        pose_afm_world = concat_transform([_create_translate(p1), inv_pose_afm])

        pose = np.argwhere(pool_volume > 0.5)
        transform_pose = apply_transform(pose, inv_pose_afm)
        transformed_pts = apply_transform(warps, afm)
        warp_vol = trilinear_interpolation(pool_volume, transformed_pts, method=method)

        reshape_vol = np.reshape(warp_vol > .5, isize).astype(np.int) * labels[i]
        resample_pose = np.argwhere(reshape_vol > 0)

        edgemap = image_utils.toedgemap(reshape_vol)
        edgemask = edgemap > 0
        edgepose = np.argwhere(edgemask)

        pose_dict[labels[i]] = edgepose + p1

        if return_normal:
            warp_vol_reshape = warp_vol.reshape(isize)
            grad_image = 1 - warp_vol_reshape

            s1 = sobel(grad_image, axis=0)
            s2 = sobel(grad_image, axis=1)
            s3 = sobel(grad_image, axis=2)

            grad = np.stack([s1, s2, s3], axis=-1)
            abs_grad = np.linalg.norm(grad, axis=-1)
            norm_grad = np.zeros_like(grad)
            nonzero_grad = abs_grad > 0.
            norm_grad[nonzero_grad] = grad[nonzero_grad] / np.expand_dims(abs_grad[nonzero_grad], axis=-1)

            normal_dict[labels[i]] = norm_grad[edgemask]

            pose_dict[labels[i]] = np.concatenate([pose_dict[labels[i]], norm_grad[edgemask]], axis=-1)
            # vec_norm = vtk_utils.create_vector(edge_grad, edgepose, scale=10, invert=True)

        crop = full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx]
        crop = np.where(crop == 0, reshape_vol, crop)
        full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx] = crop

        if DEBUG:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            pts = transform_pose.T
            pts2 = resample_pose.T
            ax.plot(*pts, 'r*')
            ax.plot(*pts2, 'g*')

    # pose_list = np.concatenate(pose_list, axis=0)
    if DEBUG:
        vtk_utils.show_actors([*vtk_utils.auto_refinement_mask(full_refinement_mask)])
        refine_pose = np.argwhere(full_refinement_mask > 0)
        reifne_pose_vtk = vtk_utils.create_points_actor(refine_pose, invert=True)
        vols = vtk_utils.numpyvolume2vtkvolume(full_refinement_mask * 255, 123)
        # vtk_utils.show_actors([reifne_pose_vtk, vtk_utils.create_points_actor(pose_list, invert=True)])
        vtk_utils.show_actors([reifne_pose_vtk, ])
        # show_plots(aligned_pts)
    outs = [pose_dict, full_refinement_mask]
    if return_normal:
        outs.append(normal_dict)
    return outs



def decompose_basis_from_pca(pose):
    """
    from point-cloud, compute origin(center) and orthonormal basis,
    4x4matrix is compose of origin and basis
    """
    pca = PCA()
    pca.fit(pose)

    rev = np.ones([3]) * (-1)
    rev[0] = -1

    def _norm_vec(vec):
        vec[:] = vec[:] / np.linalg.norm(vec)

    def _max_args_sign(vec):
        return np.sign(vec[np.argmax(np.abs(vec))])

    i = np.argmax(np.abs(pca.components_)[0])
    # j = np.argmax(np.abs(pca.components_)[:, 1])
    # k = np.argmax(np.abs(pca.components_)[:, 2])
    basis = pca.components_.copy()


    if np.sign(pca.components_[0][0]) < 0:
    # if _max_args_sign(pca.components_[0]) < 0:
        basis[0] = -basis[0]


    temp = np.array([0, 1, 0])
    z_axis = basis[0]
    x_axis = np.cross(temp, z_axis)
    _norm_vec(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    if np.sign(x_axis[-1])< 0:
        x_axis = -x_axis

    def _check_cross(vx, vy, vz):
        return np.sign(np.dot(np.cross(vx[::-1], vy[::-1]), vz[::-1]))

    if np.sign(y_axis[1]) < 0:
        if _check_cross(x_axis, y_axis, z_axis) <0:
            basis = np.stack([z_axis, -y_axis, x_axis], axis=0)
    else:
        basis = np.stack([z_axis, y_axis, x_axis], axis=0)

    # print(_check_cross(basis[2], basis[1], basis[0]))

            # basis2 = np.stack([basis[0], -y_axis, -x_axis], axis=0)

    assert np.abs(np.dot(basis, basis.T) - np.eye(3)).sum() < 1e-6

    ixpair = np.expand_dims(np.arange(3), axis=0) + np.expand_dims(np.arange(3), axis=-1)
    ixpair = ixpair % 3
    mapper = np.array([2, 1, 0])
    ixpair = mapper[ixpair]
    # ixpair[:, :2] = np.sort(ixpair[:, :2], axis=-1)[:, ::-1]
    check = []
    for pair in ixpair:
        i, j, k = pair        # print(pair)
        sign = _check_cross(basis[i], basis[j], basis[k])
        check.append(np.sign(sign))

    assert np.all(np.array(check)>0), "check orthonormal basis"



    transorom_pose = np.dot(pose - pca.mean_, basis.T)  # pca.transform(pose)
    mat = create_transform(basis, pca.mean_)
    # np.isclose(vtk_utils.apply_trasnform_np(pose, mat), transorom_pose)
    return mat



def create_transform(rot, trans):
    basisT = rot
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basisT
    transform_mat[:3, 3] = -np.dot(basisT, trans)
    return transform_mat


def _refinement_mask(resample_volume, obb_meta, source_shape, method="nearest", labels=None, threshold=0.5):
    """
    :param resample_volume: [N, d, h, w]
    :param obb_meta: origin of orthonormal basis [N, 3]
    :param source_shape: to refinement shape of volume, tuple of [3]
    :return: [source_shape],[D,H,W] restored full-mask
    """

    # pool_shape = np.array([v.shape for v in resample_volume])
    bases, centers, fsizes, theta = split_meta(obb_meta)

    # [num_rois, 8], compute extent of obb in aabb-domain
    obb_vertices = create_obb_pose_array(centers, fsizes, bases)

    max_inds = np.asarray(source_shape) - 1
    offset = 1

    num_detections = obb_meta.shape[0]

    full_refinement_mask = np.zeros([*source_shape, num_detections], dtype=np.uint8)

    labels = np.ones([obb_meta.shape[0]], dtype=np.int) if labels is None else labels
    bboxes = []

    def _create_translate(translation):
        dim = translation.size
        mat = np.eye(dim + 1)
        mat[:dim, dim] = translation
        return mat

    for i, (pts, pool_volume, origin, basis, fsize) in enumerate(
            zip(obb_vertices, resample_volume, centers, bases, fsizes)):
        # bbox =  np.concatenate(np.min(pts, axis=0), np.max(pts, axis=0), axis=0)

        p1 = np.floor(np.clip(np.min(pts, axis=0) - offset, 0, max_inds)).astype(np.int)
        p2 = np.ceil(np.clip(np.max(pts, axis=0) + offset, 0, max_inds)).astype(np.int)
        z1, y1, x1 = p1
        z2, y2, x2 = p2
        isize = p2 - p1 + 1
        dz, dy, dx = isize
        resample_scale = (np.asarray(pool_volume.shape) - 1) / fsize

        wz = np.linspace(z1, z2, dz)
        wy = np.linspace(y1, y2, dy)
        wx = np.linspace(x1, x2, dx)

        warps = np.stack(np.meshgrid(wz, wy, wx, indexing='ij'), axis=-1).reshape([-1, 3])

        # 1. scaling
        sm = np.diag(np.concatenate([resample_scale, [1]]))

        # obb-coordinate
        Tmat = create_basis_transform(basis, origin)

        # translate to origin of image-coordinate(obb-samples)
        biases = fsize / 2
        bt = _create_translate(biases)

        # concatenate transform
        afm = np.matmul(sm, np.matmul(bt, Tmat))

        transformed_pts = apply_transform(warps, afm)

        # show_plots(warps)
        warp_vol = trilinear_interpolation(pool_volume, transformed_pts, method=method)
        reshape_vol = np.reshape(warp_vol > threshold, isize).astype(np.int) * labels[i]

        # crop = full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx]
        # crop = np.where(crop == 0, reshape_vol, crop)
        # full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx] = crop
        bbox = np.array([z1, y1, x1, z1 + dz, y1 + dy, x1 + dx])
        bboxes.append(bbox)
        full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx, i] = reshape_vol

    bboxes = np.array(bboxes)
    return full_refinement_mask, bboxes


def refinement_mask(resample_volume, obb_meta, source_shape, method="nearest", labels=None):
    """
    :param resample_volume: [N, d, h, w]
    :param obb_meta: origin of orthonormal basis [N, 3]
    :param source_shape: to refinement shape of volume, tuple of [3]
    :return: [source_shape],[D,H,W] restored full-mask
    """

    # pool_shape = np.array([v.shape for v in resample_volume])
    bases, centers, fsizes, theta = split_meta(obb_meta)

    # [num_rois, 8], compute extent of obb in aabb-domain
    obb_vertices = create_obb_pose_array(centers, fsizes, bases)

    max_inds = np.asarray(source_shape) - 1
    offset = 1

    full_refinement_mask = np.zeros(source_shape, dtype=np.uint8)

    labels = np.ones([obb_meta.shape[0]], dtype=np.int) if labels is None else labels

    def _create_translate(translation):
        dim = translation.size
        mat = np.eye(dim + 1)
        mat[:dim, dim] = translation
        return mat

    for i, (pts, pool_volume, origin, basis, fsize) in enumerate(
            zip(obb_vertices, resample_volume, centers, bases, fsizes)):
        # bbox =  np.concatenate(np.min(pts, axis=0), np.max(pts, axis=0), axis=0)

        p1 = np.floor(np.clip(np.min(pts, axis=0) - offset, 0, max_inds)).astype(np.int)
        p2 = np.ceil(np.clip(np.max(pts, axis=0) + offset, 0, max_inds)).astype(np.int)
        z1, y1, x1 = p1
        z2, y2, x2 = p2
        isize = p2 - p1 + 1
        dz, dy, dx = isize
        resample_scale = (np.asarray(pool_volume.shape) - 1) / fsize

        wz = np.linspace(z1, z2, dz)
        wy = np.linspace(y1, y2, dy)
        wx = np.linspace(x1, x2, dx)

        warps = np.stack(np.meshgrid(wz, wy, wx, indexing='ij'), axis=-1).reshape([-1, 3])

        # 1. scaling
        sm = np.diag(np.concatenate([resample_scale, [1]]))

        # obb-coordinate
        Tmat = create_basis_transform(basis, origin)

        # translate to origin of image-coordinate(obb-samples)
        biases = fsize / 2
        bt = _create_translate(biases)

        # concatenate transform
        afm = np.matmul(sm, np.matmul(bt, Tmat))

        transformed_pts = apply_transform(warps, afm)

        # show_plots(warps)
        warp_vol = trilinear_interpolation(pool_volume, transformed_pts, method=method)
        reshape_vol = np.reshape(warp_vol > .5, isize).astype(np.int) * labels[i]

        crop = full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx]
        crop = np.where(crop == 0, reshape_vol, crop)
        full_refinement_mask[z1:z1 + dz, y1:y1 + dy, x1:x1 + dx] = crop

    return full_refinement_mask


def trilinear_interpolation(volume, pts, method="linear"):
    d, h, w = volume.shape
    mz = np.linspace(0, d - 1, d)
    my = np.linspace(0, h - 1, h)
    mx = np.linspace(0, w - 1, w)

    source_interpolation = RegularGridInterpolator((mz, my, mx),
                                                   volume,
                                                   method=method,
                                                   bounds_error=False,
                                                   fill_value=0)

    return source_interpolation(pts)


def gen_random_obb(bounding_box, shift_weights=[0.1, 0.05, 0.05], orientation_weights=[np.pi / 15, np.pi / 20, 0]):
    """
    :param boundary: zyx size axis-aligned bounding box [6]
    :param shift_weights:
    :param orientation_weights:
    :return:[9] float obb-meta [pose_zyx, size_zyx, euler_theta_zyx], modified
    """

    shift_weights = np.asarray(shift_weights)
    orientation_weights = np.asarray(orientation_weights)
    # split 2-vertices of bounding-box
    p1, p2 = bounding_box[:3], bounding_box[3:]
    # random crop-obb
    boundary_fsize = p2 - p1
    fsize = boundary_fsize
    # fullmask.shape
    shift_weights = np.asarray(shift_weights)
    random_shift_weights = np.random.uniform(0, shift_weights)
    # random shift
    random_ctr = (p1 + p2) / 2. + random_shift_weights * boundary_fsize

    random_theta = np.random.uniform(0, orientation_weights)
    obb = np.concatenate([random_ctr, fsize, random_theta])
    return obb


def test_create_matrix_from_euler():
    t = vtk_utils.myTransform()
    t.RotateWXYZ(30, 1, 3, 2)
    mat = t.convert_np_mat()
    deg = t.GetOrientation()
    theta = decompose_matix2euler2(mat)
    print("---", np.rad2deg(theta[::-1]))
    print("---", deg)

    wxyz = t.GetOrientationWXYZ()
    tt = vtk_utils.myTransform()
    tt.RotateWXYZ(*wxyz)
    np.isclose(mat, tt.convert_np_mat())

    mat2 = create_matrix_from_euler(theta)
    print("---", np.isclose(mat, mat2))
    # np.isclose(create_matrix_from_euler(np.deg2rad(deg[::-1])), mat)

    pose = np.random.randn(3) * 20
    size = np.abs(np.random.normal(20, 4, [3]))
    theta = np.random.uniform(-np.pi / 10, -np.pi / 10, [3])
    obb = np.concatenate([pose, size, theta])
    mat4x4 = create_matrix_from_euler(theta)
    mat4x4[:3, 3] = -np.dot(mat4x4[:3, :3], pose)
    tobb = transform_obb(mat4x4, obb[np.newaxis])
    print(tobb)


def gen_mat():
    t = vtk_utils.myTransform()
    scale = [np.random.uniform(0.8, 1.2)]*3
    t.Scale(*scale)
    t.RotateWXYZ(np.random.uniform(-np.pi, np.pi), *tuple(np.random.randn(3)))
    t.Translate(*tuple(np.random.uniform(-10, 10, [3])))
    return t.convert_np_mat()


def get_obb():
    pose = np.random.uniform(-10, 10, [3])
    size = np.random.uniform(10, 20, [3])
    tehta = np.random.uniform(-np.pi / 20, np.pi / 5, [3])
    return np.concatenate([pose, size, tehta])


def positive_scales(basis):
    """
    3x3 rotation matrix 재정렬 basis = [x;y;z] x, y, z 는 1x3 row-vector
    x, y, z 순서로 major axis가 되도록 변경
    right hand rule 이 되도록 조정( x X y = y . sklearn pca 로 주성분 분석결과가 right-hand rule 되도록 basis가 생성되지 않는다.)
    Parameters
    ----------
    basis : np.ndarray
        3x3 matrix
    Returns
    -------
        new_basis : np.ndarray
            new basis 3x3

    """
    z_axis, y_axis, x_axis = [np.squeeze(p) for p in np.split(basis, 3, axis=0)]
    # y_axis = np.cross(z_axis, x_axis)
    if np.sign(z_axis[np.argmax(np.abs(z_axis))]) < 0:
        z_axis = -z_axis

    def _check_cross(vz, vy, vx):
        return np.sign(np.dot(np.cross(vz, vy), vx))

    if _check_cross(z_axis, y_axis, x_axis) < 0:
        y_axis = -y_axis

    new_basis = np.stack([z_axis, y_axis, x_axis], axis=0)
    return new_basis


def align_transform(points):
    """
    pca 분석을 해서 alignoemnt 하는 변환행렬을 계산한다.
    alignemnt는 변환했을 때 (center ( sum(points) / #N )가 origin으로,  mojo 주성분이 x축이 되도록 처리한다.
    Parameters
    ----------
    points : np.ndarray
        (N, 3) points


    Returns
    -------

    """
    pca = PCA().fit(points)

    # normals = vtk_utils.numpy_support.vtk_to_numpy(poly.GetPointData().GetNormals())
    # pca align
    r0 = positive_scales(pca.components_)

    return create_transform(r0, pca.mean_)


def obbmeta_from_pose(align_pose, align_matrix: np.ndarray):
    # rot = t0[:3, :3]
    rot = align_matrix[:3, :3]
    # rot = np.linalg.inv(align_matrix[:3, :3])
    # rot = np.linalg.inv(rot)
    transpform = Rotation.from_matrix(rot)

    assert np.allclose(transpform.as_matrix(), rot)

    euelr_mat = Rotation.from_euler('xyz', transpform.as_euler('xyz')).as_matrix()
    assert np.allclose(euelr_mat, rot)

    euler_theta = transpform.as_euler('xyz')

    b0, b1 = align_pose.min(axis=0), align_pose.max(axis=0)

    ctr = (b0 + b1) / 2
    size = b1 - b0
    # restore center
    t_ctr = vtk_utils.apply_trasnform_np(ctr, np.linalg.inv(align_matrix))

    return np.concatenate([t_ctr, size, euler_theta])



def compute_obb_meta(points, extend_size=1.0):
    """
    from (N, 3) points. compute obb-meta by PCA

    Parameters
    ----------
    points : np.ndarray
        (N, 3)

    Returns
    -------
        (9,) obb-meta

    """
    align_t = align_transform(points)
    align_points = apply_transform(points, align_t)
    meta = obbmeta_from_pose(align_points, align_t)
    # 나도 이쯤 되면 헷갈린다...
    theta = meta[6:9]
    # size off
    meta[3:6] = meta[3:6] * extend_size
    meta[6:9] = theta[::-1]
    return meta



def inverse_theta(theta):
    """
    inv_theta = inverse_theta(theta)
    Rotate(theta) == inverse(Rotation(inv_theta)
    Parameters
    ----------
    theta : np.ndarray

    Returns
    -------
        inv_theta: np.ndarray
            inversed theta


    """
    inv_tehta = Rotation.from_matrix(Rotation.from_euler('xyz', theta).as_matrix().T).as_euler('xyz')
    return inv_tehta


def norm_boxes(boxes, shape):
    scale = np.asarray(shape)
    scale = np.pad(scale, [0,3], mode='wrap')
    # scale = scale.repeat(1, 2)

    # return  np.divide(boxes, scale).astype(np.float32)
    return np.divide(boxes, scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    scale = np.asarray(shape)
    scale = np.pad(scale, [0,3], mode='wrap')
    # scale = scale.repeat(1, 2)
    # return boxes * scale
    return np.multiply(boxes, scale)


if __name__ == "__main__":
    pass
    # test_transform_obb()
    # test_alignment_obbmeta()
    # test_create_matrix_from_euler()
    # 1. compare random orthonormal basis, and generating euler angle from rotation matrix(orthonormal baiss)
    # and then, restore rotation matrix from decomposed euler angle
    # check the results
    # test_random_vec()

    # test_gen_visualize()

    # test_compute_bounds()

    # test_gen_db()
    # test_obb_voi_align_refinement()

    # test_obb_visualize()
    # test_voi_align_obb()

    # test_concat_bbox_regressor()

    # test_volumedata()

    # x1 = tf.random_uniform([1, 9])
    # x2 = tf.random_uniform([20, 9])
    # iou = compute_obb_regressor_loss_overlaps_graph(x1, x2)[0]
    # print(iou)


