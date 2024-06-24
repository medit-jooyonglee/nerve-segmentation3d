import pyInterpolator as py
from typing import List
import numpy as np

def test_volume_merge():
    dtype = 'float32'
    num = 3
    x = np.random.randn(5, 5, 3).astype(dtype)
    # y = np.random.randn(3, 2, 3, 2).astype(dtype)
    shape = np.array([2, 3, 2])

    y = []
    for i in range(num):
        y.append(np.random.randn(2, 3, 2).astype(dtype))

    ys = np.stack(y, axis=0)
    print(ys.ravel())
    p = np.zeros([3, 6]).astype('int32')
    p[:, 3:] = shape
    # p =  np.rando
    # print(p)
    merge = py.VolumeMerge()

    # # print(x)
    # print(x.strides, x.itemsize)
    # print(x.ravel())
    copyx = x.copy()
    x0 = np.zeros_like(x)
    merge.merge(x0, y, p)
    assert (np.max(np.argwhere(x0), axis=0) == (shape-1)).all()
    i, j, k = shape
    assert np.isclose(x0[:i, :j, :k], np.mean(y, axis=0)).all()


def merge_volume_no_scale(split_vols:List[np.ndim], bboxes:np.ndarray, full_shape):
    assert len(split_vols) == bboxes.shape[0]
    counts = np.zeros(full_shape, dtype=split_vols[0].dtype)
    outs = np.zeros(full_shape, dtype=split_vols[0].dtype)

    bbox_min, bbox_max = bboxes[:, :3], bboxes[:, 3:]
    w0 = np.ones_like(split_vols[0])
    for vol, b1, b2 in zip(split_vols, bbox_min, bbox_max):
        i1, i2, i3 = b1
        j1, j2, j3 = b2
        b2_clip = np.minimum(b2, full_shape)
        c1, c2, c3 = b2_clip
        diff_shape = b2_clip - b1
        if (vol.shape == diff_shape).all():
            # outs[i1:]
            outs[i1:j1, i2:j2, i3:j3] += vol
            counts[i1:j1, i2:j2, i3:j3] += w0
        else:
            i, j, k = diff_shape
            sel_vol = vol[:i, :j, :k]
            outs[i1:c1, i2:c2, i3:c3] += sel_vol
            counts[i1:c1, i2:c2, i3:c3] += np.ones_like(sel_vol)

    trim_counts = np.where(counts > 0, counts, np.ones_like(counts))
    return outs / trim_counts


def merge_volume_no_scale_binding(split_vols:List[np.ndim], bboxes:np.ndarray, full_shape):
    assert len(split_vols) == bboxes.shape[0]
    # counts = np.zeros(full_shape, dtype=split_vols[0].dtype)
    outs = np.zeros(full_shape, dtype=split_vols[0].dtype)


    merge = py.VolumeMerge()
    merge.merge(outs, split_vols, bboxes)
    return outs


def get_slidling_box(volume_shape, box_shape, sliding):
    """
    정수형 박스 범위를 가져온다.
    정수형 박스? [h, w] image일 경우 전체를 포함하는 박스는/ [h, w] 이다.


    Parameters
    ----------
    volume_shape :
    box_shape :
    sliding :

    Returns
    -------

    """
    volume_shape = np.asarray(volume_shape)
    ndim = volume_shape.size
    # compute splited bboxes
    full_size = volume_shape - 1
    step = box_shape - sliding
    div = (full_size // step) + 1


    cover_size = step * ( div - 1) + box_shape
    print('division', div)
    print('conver', cover_size)
    print('step', step)

    assert (full_size < cover_size).all()
    # print(div)

    batch = []
    # creating bboxes
    for i in range(ndim):
        j = np.arange(0,  (step * div)[i], step[i])
        batch.append(j)
    res = np.meshgrid(*batch, indexing='ij')
    # print(res)
    pose = np.stack(res, axis=-1)


    reshape_pose = pose.reshape([-1, 3])


    bbox_min = reshape_pose
    bbox_max = reshape_pose + box_shape
    return bbox_min, bbox_max

def split_volume_no_scale(volume:np.ndarray, shape:np.ndarray, sliding:np.ndarray):

    bbox_min, bbox_max = get_slidling_box(volume.shape, shape, sliding)

    split_vols = []
    for b1, b2 in zip(bbox_min, bbox_max):
        i1, i2, i3 = b1
        j1, j2, j3 = b2
        vol = volume[i1:j1, i2:j2, i3:j3]
        # print(vol.shape, shape, vol.shape == shape)
        if (vol.shape == shape).all():
            pass
        else:
            temp = np.zeros(shape, volume.dtype)
            i, j, k = vol.shape
            temp[:i, :j, :k] = vol
            vol = temp
        assert (vol.shape == shape).all()
        split_vols.append(vol)

    split_boxes = np.concatenate([bbox_min, bbox_max], axis=-1)

    return split_vols, split_boxes

def test_volume_merge_out_of_bound():
    dtype = 'float32'
    num = 3
    x = np.random.randn(5, 5, 3).astype(dtype)
    # y = np.random.randn(3, 2, 3, 2).astype(dtype)
    shape = np.array([2, 3, 2])

    y = []
    for i in range(num):
        y.append(np.random.randn(2, 3, 2).astype(dtype))

    p = np.zeros([3, 6]).astype('int32')
    p[:, :3] = 4
    p[:, 3:] = p[:, :3] + shape
    # p =  np.rando
    # print(p)
    merge = py.VolumeMerge()

    res2 = merge_volume_no_scale(y, p, x.shape)

    res1 = merge_volume_no_scale_binding(y, p, x.shape)

    assert np.isclose(res1, res2).all()

#
# test_volume_merge_out_of_bound()
# test_volume_merge()


def test_split_merge_no_scale():

    vol_shape = [300, 400, 200]
    patch_shape = [128, 128, 128]
    # sliding = [10, 10, 10]
    sliding = [0, 0, 0]
    volume = np.random.randn(*vol_shape).astype(dtype='float32')
    shape = np.array(patch_shape)
    sliding = np.array(sliding)

    split_vols, bboxes = split_volume_no_scale(volume, shape, sliding)

    assert np.all([np.shape(v) == shape for v in split_vols])
    # split_vols2 = np.asarray(split_vols)
    merged_vol = merge_volume_no_scale(split_vols, bboxes, volume.shape)
    merged_vol2 = merge_volume_no_scale_binding(split_vols, bboxes, volume.shape)
    abs_diff = np.abs(merged_vol - merged_vol2)
    # np.sort(abs_diff)
    tol = 1e-4
    size = np.argwhere(abs_diff > tol)
    print('diff size', size.shape[0] / abs_diff.size, '//', size.shape)

    assert volume.shape == merged_vol.shape
    assert np.isclose(volume, merged_vol).all()
    # assert np.isclose(volume, merged_vol2).all()
    assert (abs_diff < tol).all(), 'max-val{}'.format(np.max(abs_diff))

test_split_merge_no_scale()