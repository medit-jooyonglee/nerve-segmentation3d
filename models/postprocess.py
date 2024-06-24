import numpy as np
import cc3d

from skimage import morphology
import rdp
from scipy import interpolate

from tools import vtk_utils, image_utils
try:
    from tools import pymeshlibs as msl
except Exception as e:
    print(f'cannot use binding module meshlibs')
from commons import timefn2, timefn
from scipy.signal import savgol_filter
from commons import get_runtime_logger

# 디버깅. 임시용 후처리 기능 예외처리
g_debug_postprocess = True


def search_block_outside_connected_components(skeleton, center_point, radius=2):
    block_outside = skeleton.copy()
    block_outside = block_outside[center_point[0] - radius:center_point[0] + radius + 1,
         center_point[1] - radius:center_point[1] + radius + 1,
         center_point[2] - radius:center_point[2] + radius + 1]
    block_outside[1:-1, 1:-1, 1:-1] = 0
    cc_labels = cc3d.connected_components(block_outside)

    return cc_labels


def find_block_coordinates(center_point, array_size, radius=2):
    if len(center_point) != 3:
        raise ValueError("Input should be a 3D point with three coordinates.")

    ranges = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid([-radius, radius], ranges, ranges, indexing='ij')
    grid_coords = np.concatenate([np.stack((x, y, z), axis=-1).reshape(-1, 3)])

    x, y, z = np.meshgrid(ranges, [-radius, radius], ranges, indexing='ij')
    grid_coords = np.concatenate([grid_coords, np.stack((x, y, z), axis=-1).reshape(-1, 3)])

    x, y, z = np.meshgrid(ranges, ranges, [-radius, radius], indexing='ij')
    grid_coords = np.concatenate([grid_coords, np.stack((x, y, z), axis=-1).reshape(-1, 3)])

    coordinates = center_point + grid_coords

    coordinates[coordinates < 0] = 0
    coordinates[coordinates[:, 0] >= array_size[0]] = array_size[0] - 1
    coordinates[coordinates[:, 1] >= array_size[1]] = array_size[1] - 1
    coordinates[coordinates[:, 2] >= array_size[2]] = array_size[2] - 1

    if not coordinates.shape[0] == (radius * 2 + 1) ** 2 * 6:
        raise ValueError("Invalid number of coordinates")
    if ((coordinates < 0).any() or (coordinates[:, 0] >= array_size[0]).any()
            or (coordinates[:, 1] >= array_size[1]).any() or (coordinates[:, 2] >= array_size[2]).any()):
        raise ValueError("Out of volume array size")

    return coordinates

@timefn
def prune_skeleton(skeleton):
    skel_labels = cc3d.connected_components(skeleton)
    new_skel = np.zeros_like(skeleton)

    for j in range(1, skel_labels.max()+1):
        temp_skel = np.zeros_like(skeleton)
        ix, iy, iz = np.where(skel_labels == j)
        temp_skel[ix, iy, iz] = 1
        skel_coords = np.stack([ix, iy, iz], axis=-1)
        branch_points = []

        # 끊긴 신경관의 경우 threshold 값 미만일 경우 유지
        if skel_coords.shape[0] < 20:
            continue

        # Extract branch points
        for i in range(skel_coords.shape[0]):
            cp = skel_coords[i, :]

            # block_coords = find_block_coordinates(cp, skeleton.shape, radius=2)
            # block_coords = np.unique(block_coords, axis=0)
            # X, Y, Z = list(block_coords.T)
            # point_count = np.count_nonzero(skeleton[X, Y, Z])
            # if point_count >= 3:
            #     branch_points.append(cp)

            cc_labels = search_block_outside_connected_components(temp_skel, cp, radius=2)

            if cc_labels.max() >= 3:
                branch_points.append(cp)

        if len(branch_points) == 0:
            ix, iy, iz = np.nonzero(temp_skel)
            new_skel[ix, iy, iz] = 255
        else:
            # skeleton에서 branch points 제거
            X, Y, Z = list(np.array(branch_points).T)
            temp_skel[X, Y, Z] = 0

            # Removal of small objects ("dust") using 3d connected components
            pruned_skel = cc3d.dust(temp_skel, threshold=15, connectivity=26, in_place=False)

            ix, iy, iz = np.nonzero(pruned_skel)
            new_skel[ix, iy, iz] = 255

    return new_skel


# @timefn2
def sorting_by_distances(points):
    dist = np.linalg.norm(points[np.newaxis] - points[:, np.newaxis], axis=-1)
    remain = points
    remain_dist = dist
    size = remain_dist.shape[0]
    remain_dist[np.arange(size), np.arange(size)] = 1e10


    def next_arg(v, i):
        order = np.arange(v.shape[0])
        arg = np.concatenate([order[:i], order[i + 1:]])
        return arg

    next_i = 0
    collected = [
        remain[next_i]
    ]
    arg = next_arg(remain, next_i)

    next_i = np.argmin(remain_dist[next_i][arg])
    remain = remain[arg]
    remain_dist = remain_dist[arg][:, arg]

    # def sorting_by_distance():
    while remain.size > 0:
        # print(remain.size)
        collected.append(remain[next_i])
        if remain.shape[0] == 1:
            break
        arg = next_arg(remain, next_i)
        next_i = np.argmin(remain_dist[next_i][arg])
        remain = remain[arg]
        # if remain.size == 0:
        #     break
        remain_dist = remain_dist[arg][:, arg]

    collected = np.stack(collected, axis=0)
    return collected

# from skimage.measure import LineModelND, ransac

def smoothing_spline(points, num=None):
    """
    # centers = pts / norm_term
    # https://stackoverflow.com/questions/18962175/spline-interpolation-coefficients-of-a-line-curve-in-3d-space

    Parameters
    ----------
    points :
    num : int number of spline-points

    Returns
    -------

    """
    s = None
    tck, u = interpolate.splprep([points[:, i] for i in range(points.shape[-1])], s=s)

    n = num or points.shape[0]
    u_fine = np.linspace(0, 1, n)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    spline_points = np.stack([x_fine, y_fine, z_fine], axis=1)
    return spline_points
#
# class SplineModelND(LineModelND):
#     dist_volume = np.ndarray([])
#     def __init__(self):
#         pass
#
#
#     def estimate(self, data):
#         """
#         계산 자체가 필요없는것 같다. residual값만 생성하면 된다.
#         Parameters
#         ----------
#         data :
#
#         Returns
#         -------
#
#         """
#         return True
#         # s = None
#         # tck, u = interpolate.splprep([data[:, 0], data[:, 1], data[:, 2]], s=s)
#         # # x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
#         # u_fine = np.linspace(0, 1, N)
#         # x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
#         # spline_points = np.stack([x_fine, y_fine, z_fine], axis=1)
#         # return
#
#     def residuals(self, data, params=None):
#         idata = data.astype(np.int32)
#         i, j, k = idata[:, 0],  idata[:, 1],  idata[:, 2]
#         return SplineModelND.dist_volume[i, j, k]


# @timefn2
def segmentmask_to_spline(mask, small_obj=8**3, visualize=False, order='zyx', remove_outlier=False, return_spline_points=False) -> np.ndarray:
    """
    (d, h, w) volume-mask로부터 spline 좌표점을 생성한다.
    1. noise-remove - remove small object(noise-remover)
    2. skeletonize
    3. coords sampling and sorting :
    4. sorting coordinates : using nearest-neighbor
    5. savistsky-golume & ransac
    6. spline compute
    7. sampling - major control-points

    Parameters
    ----------
    mask :
    visualize : bool default False. debug option to visualize

    Returns
    -------

    """
    # FIXME: we need to set small-size by spacing
    logger = get_runtime_logger()
    # small_obj = 5**3  # (5, 5, 5)
    post_mask = morphology.remove_small_objects(mask > 0, min_size=small_obj)
    # bin_tooth_label = morphology.remove_small_objects((labels == tooth_id), 500)
    fill_mask = morphology.closing(post_mask)
    skel = morphology.skeletonize_3d(fill_mask)
    if not skel.max() == 0:
        pruned_skel = prune_skeleton(skel)
    else:
        pruned_skel = skel
    coords = np.stack(np.nonzero(pruned_skel), axis=-1)

    if coords.shape[0] < 3:
        logger.error('cannot post-processing for spline. too small coordinate:{}'.format(coords.shape))
        return coords

    # first sorting z order in increasing
    iz = np.argsort(coords[:, 0])
    sort_coords = coords[iz]
    #
    compute_distance_transform = False
    if compute_distance_transform:
        edt = msl.PyEdt(mask.astype(np.uint8))
        edt.update()
        dist_vol = edt.distance_array.reshape(mask.shape)

        max_dist = dist_vol.max()
        outout_dist = np.full_like(dist_vol, dist_vol.max() * 5)

        # 가장자리 부분에 distance값이 높도록 처리
        inv_dist_vol = max_dist - dist_vol
        dist_vol = np.where(mask, inv_dist_vol, outout_dist)


    sorted_seq_points2 = sorting_by_distances(sort_coords[::2])

    # TODO: sorting 결과 small object(찌꺼기같은)가 남아서 spline 처리가 꼬여버리는 문제.
    #       임시로 거리로 산출해서 뒷부분 데이터가 역으로 점핑하는 경우 해당 데이터를 버린다.
    #       실험값으로 포인트간에 거리는 2~3 정도. 표준편차는 1.5~2
    #       sorted_seq_points2[:, 0]

    truncated_sorted_seq_points2 = sorted_seq_points2
    if g_debug_postprocess:
        logger.info('g_debug_postprocessing...')
        diff_dist = np.linalg.norm(np.diff(sorted_seq_points2, axis=0), axis=1)
        z_len = mask.shape[0]
        diff_dist_mask = np.zeros_like(diff_dist, dtype=np.bool_)
        diff_dist_mask[int(diff_dist_mask.size * 0.6):] = True
        invalid_args = np.where(np.logical_and(diff_dist > z_len / 3, diff_dist_mask))[0]

        if invalid_args.size > 0:

            invalid_arg = np.min(invalid_args)
            # differential 기준으로 1`더해준다.
            truncated_sorted_seq_points2 = sorted_seq_points2[:invalid_arg+1]
        else:
            truncated_sorted_seq_points2 = sorted_seq_points2

    n = truncated_sorted_seq_points2.shape[0]
    N = np.maximum(n // 100 + 1, 1) * 100



    smooth_spline_points2 = savgol_filter(truncated_sorted_seq_points2, 7, 3, axis=0)
    # spline_points = smoothing_spline(truncated_sorted_seq_points2, num=N)

    spline_points = smooth_spline_points2

    sample_points = rdp.rdp(spline_points, epsilon=0.5)

    if visualize:
        smooth_spline_points1 = smoothing_spline(truncated_sorted_seq_points2, num=N)

        point_size = 3
        vtk_utils.split_show([
            # smooth_spline_points1,
            vtk_utils.create_points_actor(smooth_spline_points1, color_value=(1, 1, 1), point_size=point_size),
            vtk_utils.create_points_actor(truncated_sorted_seq_points2, color_value=(0, 1, 0), point_size=point_size),
        ], [
            vtk_utils.create_points_actor(spline_points, color_value=(1, 1, 1), point_size=point_size),            # smooth_spline_points2,
            vtk_utils.create_points_actor(truncated_sorted_seq_points2, color_value=(0, 1, 0), point_size=point_size)
        ], render_colors=[(0, 0, 0), (.1, .1, .1)]
        )

        vtk_utils.split_show([
            truncated_sorted_seq_points2, *vtk_utils.create_sphere(sample_points, 1.0)
        ], [
            spline_points, *vtk_utils.create_sphere(sample_points, 1.0)
        ], render_colors=[(0, 0, 0), (.1, .1, .1)]
        )

        vtk_utils.split_show([
            truncated_sorted_seq_points2, *vtk_utils.create_sphere( rdp.rdp(smooth_spline_points1, epsilon=0.5), 1.0)
        ], [
            truncated_sorted_seq_points2, *vtk_utils.create_sphere(sample_points, 1.0)
        ], render_colors=[(0, 0, 0), (.1, .1, .1)]
        )





    # sa
    if order == 'xyz':
        sample_points = sample_points[:, ::-1]

    ####
    if visualize:
        logger.info('input-mask')
        vtk_utils.show_actors([
            mask,
            vtk_utils.get_axes(100)
        ])

        logger.info('post-input-mask by removing small-object')
        vtk_utils.show_actors([
            post_mask,
            vtk_utils.get_axes(100)
        ])


        logger.info('skeltonizxed mask')
        vtk_utils.show_actors([
            skel,
            vtk_utils.get_axes(100)
        ])


        # sort_coords
        logger.info('sampling coordinates and sorting in z-axis')
        vtk_utils.show_actors([
            *vtk_utils.create_sphere(sort_coords[::-1], size=0.5),
            vtk_utils.get_axes(100)
        ])

        logger.info('sorting points nearest-neighbor')
        vtk_utils.show_actors([
            *vtk_utils.create_sphere(sorted_seq_points2[::-1], size=0.5),
            vtk_utils.get_axes(100)
        ])


        logger.info('spline points')
        vtk_utils.show_actors([
            vtk_utils.create_curve_actor(spline_points[::-1][:, ::-1], line_width=5),
            vtk_utils.get_axes(100)
        ])

        logger.info('final sampling points by Ramer-Douglas-Peucker Algorithm')

        vtk_utils.show_actors(
            [*vtk_utils.create_sphere(sample_points, size=1.5), vtk_utils.create_curve_actor(sample_points)])

    if remove_outlier:
        sample_points = remove_outlier_from_dot_product(sample_points.copy(),mask, 0.3, visualize)
        # return inlier

    if return_spline_points:
        return sample_points, spline_points
    else:
        return sample_points


def remove_outlier_from_dot_product(points, mask=None, threshold=0.5, visualize=False):
    '''
    인접 3 포인트를 이용해 두 벡터를 계산하고
    두 벡터간 내적의 절대값이 threshold 이하일 경우 (진행 방향이 크게 바뀔 경우 )
    outlier 후보로 분류하며, 신경관 특성상 검출 & 정렬된 포인트의 끝부분은 원래 휘어지므로
    내적 값이 작게 나옴
    따라서 길이의 0.8 이내의 outlier만 제거 대상으로 처리

    threshold는 실험적으로 0.3으로 적용
    신경관이 꺾이는 부분도 임의로 0.8로 설정함

    Args:
        points:
        mask:
        threshold:

    Returns:

    '''

    if not len(points) > 2:
        return points

    new_points = points.copy()

    unit_vectors = []
    outlier_idx = []

    init_vector = points[1]-points[0]
    prev_vector = init_vector

    for i in range(len(points)-2):
        cur_vector = points[i+2] - points[i+1]

        unit_prev = prev_vector / np.linalg.norm(prev_vector)
        unit_cur = cur_vector / np.linalg.norm(cur_vector)

        dot = np.abs(np.dot(unit_prev, unit_cur))

        if dot < threshold:
            outlier_idx.append(i+2)

    outlier_idx = [outlier for outlier in outlier_idx if outlier < int(len(points)*0.8)]
    # def recursive_range(points, outlier_idx):
    #     if len(outlier_idx) > 2:
    #         points = np.delete(points, np.s_[outlier_idx[-3]:outlier_idx[-2]], 0)
    #         outlier_idx = np.delete(outlier_idx, np.s_[-2:-1],0)
    #         points, outlier_idx = recursive_range(points, outlier_idx)
    #
    #     return points, outlier_idx
    # new_points, new_outlier_idx = recursive_range(points, outlier_idx)
    if len(outlier_idx)> 1:
        new_points = np.delete(points, np.s_[outlier_idx[0]:outlier_idx[-1]],0)

        if mask is not None and visualize:
            vtk_utils.show_actors([mask, *vtk_utils.create_sphere(points[outlier_idx], size=10)])
            vtk_utils.show_actors([mask, *vtk_utils.create_sphere(new_points, size=10)])

    # cnt = 0
    # for i in range(len(outlier_idx)):
    #     points = np.delete(points, outlier_idx[i]-cnt, 0)
    #     cnt += 1

    return new_points


def ransac_custom(points, dist_volume):
    max_trials = 1000
    num_trials = 0
    num_samples = points.shape[0]
    min_samples = int( num_samples * 0.8)

    compute_total_length = lambda x: np.sum(np.linalg.norm(np.diff(x, axis=0), axis=-1))

    while num_trials < max_trials:
        spl_idxs = np.random.choice(np.arange(num_samples), min_samples, replace=False)
        sel_mask = np.zeros([num_samples], dtype=bool)
        sel_mask[spl_idxs] = True

        # no samples repeat
        # spl_idxs = rng.choice(num_samples, min_samples, replace=False)
        x = np.cumsum(np.random.uniform(0, 1, [min_samples]))
        x = x / x.max() * ( num_samples - 1)
        ix = x.astype(np.int32)

        res_data = spline(points[sel_mask])

        idata = res_data.astype(np.int32)
        i, j, k = idata[:, 0],  idata[:, 1],  idata[:, 2]
        dist_scores = dist_volume[i, j, k]

        total_len = compute_total_length(res_data)
        print('scores', np.mean(dist_scores) * total_len)
        num_trials += 1
        # scores = total_len / dist_scores
        # print(num_trials, )

def rdp_test():
    x = np.random.randn(100, 3)
    arr = rdp.rdp(x, epsilon=1e-3, return_mask=True)
    print(arr)
    vtk_utils.show_actors([
        x,
    ])


# @timefn
def remove_small_objects(image, threshold, method):
    assert method in ['skimage', 'pymeshlibs']

    if method == 'skimage':
        return morphology.remove_small_objects(image.astype(np.bool_), threshold)
    elif method == 'pymeshlibs':
        assert np.issubdtype(image.dtype, np.integer)
        region_grow = msl.RegionGrow()
        region_grow.setInput(image.astype(np.uint8).ravel(), image.shape)
        region_grow.update()
        areas = region_grow.getAreas()
        res_image = region_grow.getOutputImage().reshape(image.shape)
        # zer
        area_mask = areas
        area_mask[areas < threshold] = 0
        mapping = np.arange(areas.size + 1)
        # print('num areas', areas.size)
        # to mask
        mapping[1:] = area_mask
        mappiing_image = mapping[res_image]
        return np.where(mappiing_image > 0 , image, np.zeros_like(image))
        # mapping[mapping > 0] = 1
        # return np.where(image, mapping >)

        # image[]
        # mapping[1:] = areas
        # return mapping[res_image]



# rdp_test()

