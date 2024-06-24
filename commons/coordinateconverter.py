import numpy as np

def translation_mat(translation):
    dim = translation.size
    mat = np.eye(dim + 1)
    mat[:dim, dim] = translation
    return mat


def scaling_mat(scale):
    dim = scale.size
    mat = np.eye(dim + 1)
    mat[:dim, :dim] = np.diag(scale)
    return mat


def create_trasnform(basis, origin):
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basis
    transform_mat[:3, 3] = -np.dot(basis, origin)
    return transform_mat


def rotation_mat(rot):
    dim = rot.shape[0]
    mat = np.eye(dim + 1)
    mat[:dim, :dim] = rot
    return mat


def create_transform(rot, trans):
    basisT = rot
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = basisT
    transform_mat[:3, 3] = -np.dot(basisT, trans)
    return transform_mat


def concat_transform(transform_list):
    """
    :param transform_list: list of array [4,4]
    :return:
    """
    mat = np.eye(4)
    for t in reversed(transform_list):
        mat = np.dot(t, mat)
    return mat


def getAutoPlanningTransform(source_volume_or_volume_shape, source_spacing):
    """
    get the coordinate system in AutoPlanning App.
    https://www.slicer.org/wiki/Coordinate_systems
    Auto-Planning coordinate
    in general RPS, but vtk RAI 를 사용한다. 즉, y, z reverse 처리

    vtk - LAI // general LPS
    Voxel(CBCT)- RPS - (from right towards left) - x
                       (from anterior towards posterior) - y
                       ( from inferior towards superior) - z
    Worlds - y-z permutation, as same y-x symmetric, and scaling by dicom-pixel-spacing
                 and origin is the center of voxel
                 autuplanning [CCoordinateConverter::CvtCoordVol2World] 참고

    Mesh(STL) - z-axis sign-inverting CSurfaceLoader 참고

    :param source_volume:voxel, [D,H,W] numpy array
    :param source_spacing:tuple pixel-spacing - mm/pixels
    :return:
    each xyz transform
    """
    source_volume_or_volume_shape = np.asarray(source_volume_or_volume_shape)
    if source_volume_or_volume_shape.ndim == 3:
        source_volume_shape = source_volume_or_volume_shape.shape
    elif source_volume_or_volume_shape.ndim == 1 and source_volume_or_volume_shape.size == 3:
        source_volume_shape = source_volume_or_volume_shape
    else:
        raise ValueError('invalid value:{}'.format(type(source_volume_or_volume_shape)))

    max_index = np.asarray(source_volume_shape[::-1])
    eyemat = np.eye(4)
    rot = eyemat[:3, :3]
    # LPS to LAI
    rot = np.stack([rot[0], -rot[1], -rot[2]], axis=0)
    # translate voxel origin from "right-superior-posterior"-to "right-aneterior-inferior"
    # max index --> 0 가 되도록
    trans_ct = max_index.copy() - 1
    trans_ct[0] = 0
    Tct = create_trasnform(rot, trans_ct)
    # Tct = mat

    ctrs = max_index / 2.
    scale_mat = scaling_mat(source_spacing)

    eye4x4 = np.eye(4)
    # y & z axis permutation
    permut = np.stack([eye4x4[0], eye4x4[2], eye4x4[1], eye4x4[-1]], axis=-1)
    ctr_permut = np.dot(permut[:3, :3], -ctrs)
    # = general_box.sca
    trans_mat = translation_mat(ctr_permut)
    vox2world = concat_transform([scale_mat, trans_mat, permut])

    Tct = concat_transform([Tct])
    # z-axis sign-invering
    Tmesh = scaling_mat(np.array([1, 1, -1]))
    return Tct, Tmesh, vox2world