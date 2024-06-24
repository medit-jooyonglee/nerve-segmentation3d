from tools import geometry_numpy
import numpy as np

# 3.11 version
# from typing import Self
from sklearn.decomposition import PCA

from scipy.spatial.transform import Rotation
from tools import vtk_utils


def to_right_hand_rules(basis):
    """
    right hand rule 축이 되도록 basis 축 부호 조정한다.
    z = cross(x, y)
    Parameters
    ----------
    basis :

    Returns
    -------

    """
    z_axis, y_axis, x_axis = [np.squeeze(p) for p in np.split(basis, 3, axis=0)]
    # y_axis = np.cross(z_axis, x_axis)
    if np.sign(z_axis[np.argmax(np.abs(z_axis))]) < 0:
        z_axis = -z_axis

    def _check_cross(vz, vy, vx):
        return np.sign(np.dot(np.cross(vz, vy), vx))

    if _check_cross(z_axis, y_axis, x_axis) < 0:
        y_axis = -y_axis

    basis = np.stack([z_axis, y_axis, x_axis], axis=0)
    return basis

def refinement_basis(basis):
    z_axis, y_axis, x_axis = [np.squeeze(p) for p in np.split(basis, 3, axis=0)]
    x_axis = np.array([0, 0, 1])
    y_axis = np.cross(x_axis, z_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(z_axis, y_axis)
    return np.stack([z_axis, y_axis, x_axis], axis=0)




class Obb:
    def __init__(self):
        self.center = np.array([])
        self.fsize = np.array([])
        self.theta = np.array([])

        self.order = 'zyx'
        self.angle_dtype = 'radian'

        self.debug = False

    def copy(self):
        out = Obb()
        out.center = self.center.copy()
        out.fsize = self.fsize.copy()
        out.theta = self.theta.copy()
        out.order = self.order
        out.angle_dtype = self.angle_dtype
        out.debug = self.debug
        return out

    def __repr__(self):
        msg = f'{self.center=} / {self.fsize=} / {self.theta=} '
        return msg

    def from_meta(self, meta:np.ndarray):
        self.center, self.fsize, self.theta = np.split(meta, 3)
        return self

    def to_meta(self) ->np.ndarray:
        return np.concatenate([self.center, self.fsize, self.theta])

    @staticmethod
    def create_meta(in_pose, order='zyx'):
        return Obb.create(in_pose, order=order).to_meta()

    def __eq__(self, other):
        return np.allclose(self.vertices, other.vertices)

    def clone(self):
        return Obb().from_meta(self.to_meta())

    def create(self, in_pose, order='zyx', refinement=True, sigma=-1, aabb=False):
        """

        Parameters
        ----------
        in_pose : np.ndarray,
            (N, 3) points
        order : str,
            the coordinates order. use zyx order in terms of volume. and scipy.transform 도 기본적으로 zyx order를 사용한다.
        sigma: int, -1 일 때는 bounding box 계산시 data 전체를 포함하도록 설정. 양수값 설정시 num * sigma 값으로 bounding box 설정


        Returns
        -------

        """
        assert in_pose.shape[0] >= 3, 'we needs 3d points at least 3'
        assert order in ['xyz', 'zyx']
        pose = in_pose if order == 'zyx' else in_pose[:, ::-1]
        self.order = order

        if aabb:
            basis = np.eye(3)
            temp_center = np.mean(pose, axis=0)
        else:
            pca = PCA().fit(pose)
            basis = to_right_hand_rules(pca.components_)
            basis = refinement_basis(basis) if refinement else basis
            assert np.isclose(basis.T.dot(basis), np.eye(3), atol=1e-6).all()
            temp_center = pca.mean_

        degree = self.angle_dtype == 'degree'

        # 앞에서 좌표값을 조정해주므로 기본값 zyx order로 처리
        euler_thata = Rotation.from_matrix(basis).as_euler('zyx', degrees=degree)

        rot = Rotation.from_euler('zyx', euler_thata).as_matrix()
        assert np.isclose(basis.T.dot(rot), np.eye(3), atol=1e-6).all()

        self.theta = euler_thata # if not aabb else np.zeros_like(eulur_thata)


        t0 = geometry_numpy.create_transform(basis, temp_center)

        align_pose = geometry_numpy.apply_transform(in_pose, t0)
        if sigma < 0:
            amin, amax = np.min(align_pose, axis=0), np.max(align_pose, axis=0)
        else:
            ext = sigma * np.std(align_pose, axis=0)
            amin, amax = -ext, ext
        self.fsize = amax - amin
        origin = (amax + amin) / 2.

        center = geometry_numpy.apply_transform(origin, np.linalg.inv(t0))
        self.center = center
        return self

    def scale_size(self, scale):
        self.fsize = self.fsize * scale
        return self

    @property
    def basis(self):
        degree = self.angle_dtype == 'degree'
        basis = Rotation.from_euler('zyx', self.theta, degrees=degree).as_matrix()
        return basis

    def transform(self, mat4x4):
        trans_obb_pose = geometry_numpy.apply_transform(self.vertices, mat4x4)

        obb = Obb()
        obb.create(trans_obb_pose, self.order)
        return obb

    def apply_transform(self, mat4x4:np.ndarray):
        """

        Parameters
        ----------
        mat4x4 :

        Returns
        -------

        """

        # z = np.linspace(-1, 1, 2)
        # y = np.linspace(-1, 1, 2)
        # x = np.linspace(-1, 1, 2)
        #
        # Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        # zyx_grid = np.stack([Z, Y, X], axis=-1)
        #

        # zyx_grid.reshape([-1, 3])
        """
        
    array([[-1., -1., -1.],
           [-1., -1.,  1.],
           [-1.,  1., -1.],
           [-1.,  1.,  1.],
           [ 1., -1., -1.],
           [ 1., -1.,  1.],
           [ 1.,  1., -1.],
           [ 1.,  1.,  1.]])
        """

        rot, scale, trans = geometry_numpy.decompose_complete_matrix(mat4x4)

        rotmat = np.eye(4)
        rotmat[:3, :3] = rot
        # trans_verts = vtk_utils.apply_trasnform_np(self.vertices, rotmat)
        trans_verts = vtk_utils.apply_trasnform_np(self.vertices, mat4x4)



        dx = trans_verts[1] - trans_verts[0]
        dy = trans_verts[2] - trans_verts[1]
        dx = dx / np.linalg.norm(dx)
        temp_dy = dy / np.linalg.norm(dy)
        dz = np.cross(dx, temp_dy)
        dz = dz / np.linalg.norm(dz)
        dy = np.cross(dx, dz)
        mat = np.stack([dz, dy, dx], axis=-1)

        mat = to_right_hand_rules(mat)


        verts = vtk_utils.apply_trasnform_np(self.vertices, mat4x4)
        align = geometry_numpy.create_transform(mat, np.mean(verts, axis=0))
        align_verts = vtk_utils.apply_trasnform_np(verts, align)
        # fsize =
        fsize = np.max(align_verts, axis=0) - np.min(align_verts, axis=0)
        meta = np.concatenate([np.mean(verts, axis=0), fsize, Rotation.from_matrix(mat).as_euler('zyx')])
        return Obb().from_meta(meta)


        # rotself.basis
        align_rot = rot.dot(self.basis)
        theta = Rotation.from_matrix(rot.dot(self.basis)).as_euler('zyx')

        center = vtk_utils.apply_trasnform_np(self.center, mat4x4)
        align_mat = geometry_numpy.create_transform(align_rot, center)
        vertices = vtk_utils.apply_trasnform_np(self.vertices, mat4x4)
        align_verts = vtk_utils.apply_trasnform_np(vertices, align_mat)

        # concat_mat = self.align_transform.dot(mat4x4)
        fsize = np.max(align_verts, axis=0) - np.min(vertices, axis=0)

        meta = np.concatenate([center, fsize, theta])
        return Obb().from_meta(meta)



        trans_verts = vtk_utils.apply_trasnform_np(self.vertices, mat4x4)
        trans_center = np.mean(trans_verts, axis=0)
        align_verts = vtk_utils.apply_trasnform_np(trans_verts, geometry_numpy.create_transform(self.basis, trans_center))
        fsize = np.max(align_verts, axis=0) - np.min(align_verts, axis=0)
        meta = np.concatenate([trans_center, fsize, self.theta])
        return Obb().from_meta(meta)
        # orthonor
        dx = trans_verts[1] - trans_verts[0]
        dy = trans_verts[2] - trans_verts[1]
        dx = dx / np.linalg.norm(dx)
        temp_dy = dy / np.linalg.norm(dy)
        dz = np.cross(dx, temp_dy)
        dz = dz / np.linalg.norm(dz)
        dy = np.cross(dx, dz)

        mat = np.stack([dz, dy, dx], axis=-1)
        rot = Rotation.from_matrix(mat)
        if self.debug:
            assert np.allclose(mat.dot(rot.as_matrix().T), np.eye(3))
            res = Rotation.from_euler('zyx', rot.as_euler('zyx')).as_matrix()
            assert np.allclose(mat.dot(res.T), np.eye(3))

        # np.mean(trans_verts, axis=0)
        trans_center = np.mean(trans_verts, axis=0)
        # mat4x4 = geometry_numpy.create_transform(mat, self.center)
        mat4x4 = geometry_numpy.create_transform(mat, trans_center)
        align_trans_verts = vtk_utils.apply_trasnform_np(trans_verts, mat4x4)

        fmax, fmin = np.max(align_trans_verts, axis=0), np.min(align_trans_verts, axis=0)
        fsize = fmax - fmin

        meta = np.concatenate([trans_center, fsize, rot.as_euler('zyx')])
        return Obb().from_meta(meta)




###################
        trans_verts = vtk_utils.apply_trasnform_np(self.vertices, mat4x4)
        # orthonor
        dx = trans_verts[1] - trans_verts[0]
        dy = trans_verts[2] - trans_verts[1]
        dx = dx / np.linalg.norm(dx)
        temp_dy = dy / np.linalg.norm(dy)
        dz = np.cross(dx, temp_dy)
        dz = dz / np.linalg.norm(dz)
        dy = np.cross(dx, dz)

        mat = np.stack([dz, dy, dx], axis=-1)
        rot = Rotation.from_matrix(mat)
        if self.debug:
            assert np.allclose(mat.dot(rot.as_matrix().T), np.eye(3))
            res = Rotation.from_euler('zyx', rot.as_euler('zyx')).as_matrix()
            assert np.allclose(mat.dot(res.T), np.eye(3))

        # np.mean(trans_verts, axis=0)
        trans_center = np.mean(trans_verts, axis=0)
        # mat4x4 = geometry_numpy.create_transform(mat, self.center)
        mat4x4 = geometry_numpy.create_transform(mat, trans_center)
        align_trans_verts = vtk_utils.apply_trasnform_np(trans_verts, mat4x4)

        fmax, fmin = np.max(align_trans_verts, axis=0), np.min(align_trans_verts, axis=0)
        fsize = fmax - fmin

        meta = np.concatenate([trans_center, fsize, rot.as_euler('zyx')])
        return Obb().from_meta(meta)


    def is_contain(self, points:np.ndarray):
        pass



    @property
    def align_transform(self) -> np.ndarray:
        # degree = self.angle_dtype == 'degree'
        # basis = Rotation.from_euler('zyx', self.theta, degrees=degree).as_matrix()
        return geometry_numpy.create_transform(self.basis, self.center)

    def as_vtkcube(self, color=None):
        fsize = self.fsize if self.order == 'zyx' else self.fsize[::-1]
        bbox = np.concatenate([-fsize/2, fsize/2])
        # vtk order xyz로 맞춰줘야한다.
        vtk_bbox = vtk_utils.convert_box_norm2vtk(bbox[np.newaxis], invert=True)
        cube = vtk_utils.get_cube(vtk_bbox[0])

        t0 = geometry_numpy.create_transform(self.basis, self.center)
        rev_t0 = geometry_numpy.reverse_transform(t0) if self.order == 'zyx' else t0

        inv_t0 = np.linalg.inv(rev_t0)

        cube = vtk_utils.apply_transform_actor(cube, inv_t0)

        cube.GetProperty().SetRepresentationToWireframe()
        cube.GetProperty().SetLineWidth(3)
        cube.GetProperty().LightingOff()

        if color is not None:
            vtk_utils.change_actor_color(cube, color)
        return cube

    def obb_pose(self):
        return self.create_obb_pose_array(self.center, self.fsize, self.basis)

    @property
    def vertices(self):
        return self.obb_pose()

    @staticmethod
    def create_obb_pose_array(center, fsize, basis):
        """
        create 8 vertices of cubes
        Parameters
        ----------
        center : np.ndarray (3)
        fsize : np.ndarray (3)
        basis : np.ndarray (3, 3) rotation matrix

        Returns
        -------

        """
        # [N,1]
        z = np.linspace(-1, 1, 2)
        y = np.linspace(-1, 1, 2)
        x = np.linspace(-1, 1, 2)

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        # [2, 2, 2, 3]
        zyx_grid = np.stack([Z, Y, X], axis=-1)
        bbox_grid = zyx_grid * (fsize/2)

        t0 = geometry_numpy.create_transform(basis, center)

        inv_t0 = np.linalg.inv(t0)
        bbox_grid = vtk_utils.apply_trasnform_np(bbox_grid.reshape([-1, 3]), inv_t0)

        return bbox_grid

    def pooling_points(self, pool_shape, scale=1.0, return_transform=False):
        """

        Args:
            pool_shape ():
            scale ():
            return_transform ():

        Returns:
        np.ndarray (d, h, w, 3) pooling points for obb & and pool_shape & and given scale
        np.ndarray optional(np.ndarray) transform matrix crop-shape to pooling points
        """
        d, h, w = pool_shape
        z = np.linspace(-1, 1, d)
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
        # [2, 2, 2, 3]
        zyx_grid = np.stack([Z, Y, X], axis=-1)

        unit_scale = self.fsize / 2 * scale
        bbox_grid = zyx_grid * unit_scale

        # basis = refinement_basis(self.basis) if refinement else  self.basis

        t0 = geometry_numpy.create_transform(self.basis, self.center)

        inv_t0 = np.linalg.inv(t0)
        bbox_grid = vtk_utils.apply_trasnform_np(bbox_grid.reshape([-1, 3]), inv_t0)
        bbox_grid_points = bbox_grid.reshape([*pool_shape, 3])
        if return_transform:

            s0 = geometry_numpy.scaling_mat(np.asarray(2/(np.asarray(pool_shape)-1)))
            t0 = geometry_numpy.translation_mat(np.asarray([-1, -1, -1]))
            s1 = geometry_numpy.scaling_mat(unit_scale)
            mat_to_box = geometry_numpy.concat_transform([inv_t0, s1, t0, s0])
            mat_to_src = mat_to_box
            return bbox_grid_points, mat_to_src
        else:
            return bbox_grid_points

        # return bbox_grid.reshape([*pool_shape, 3])

    def add_noise(self, shift, scale, theta):
        self.center = self.center + shift
        self.fsize = self.fsize + scale
        self.theta = self.theta + theta
        return self

def test_rotation_matrix():

    theta = np.array([0, np.pi/3, 0])
    # theta = np.array([np.pi/6, np.pi/3, 0])
    mat = geometry_numpy.create_matrix_from_euler(theta)
    print(mat)


    # rot = Rotation.from_euler(seq='xyz', angles=theta, degrees=False)
    rot = Rotation.from_euler(seq='zyx', angles=theta, degrees=False)

    print(rot.as_matrix())



    size = np.array([1, 2, 3])
    ctr = np.array([0, 0, 0])
    cube = vtk_utils.create_obb_cube(ctr, size, rot.as_matrix())
    # vtk_utils.show_actors([cube, vtk_utils.get_axes(5)])

def test_rotation_decompose():

    """

    - First angle belongs to [-180, 180] degrees (both inclusive)
    - Third angle belongs to [-180, 180] degrees (both inclusive)
    - Second angle belongs to:

        - [-90, 90] degrees if all axes are different (like xyz)
        - [0, 180] degrees if first and third axes are the same
          (like zxz)

    """
    from tools import geometry_numpy
    for i in range(10):
        t = np.random.uniform(0, np.pi, [3])
        m = Rotation.from_euler('xyz', t).as_matrix()
        t2_zyx = Rotation.from_matrix(m).as_euler('zyx')
        t2_xyz = Rotation.from_matrix(m).as_euler('xyz')

        m1 = Rotation.from_euler('zyx', t2_zyx).as_matrix()
        m2 = Rotation.from_euler('xyz', t2_xyz).as_matrix()
        is_otrhogonal = lambda a, b: np.isclose(a.T.dot(b), np.eye(a.shape[0])).all()
        assert is_otrhogonal(m1, m2)

        # r1 = Rotation.from_euler('zyx', t2_zyx).as_matrix()
        # r2 = Rotation.from_euler('xyz', t2_zyx).as_matrix()
        # r2 = geometry_numpy.reverse_rotation(r2)
        # assert is_otrhogonal(r1, r2)
        # print(t2, np.rad2deg(t2))
        print(f'input theta {t=} / {t2_zyx=} / rever {t2_xyz=}')

def test_rotation_matrix():
    theta = np.array([np.pi/3, 0, 0])
    # rot = Rotation.from_euler(seq='xyz', angles=theta, degrees=False)
    # rot = Rotation.from_euler(seq='zyx', angles=theta, degrees=False)
    rot = Rotation.from_euler(seq='zyx', angles=theta, degrees=False)

    mat_zyx = rot.as_matrix()
    assert mat_zyx[-1, -1] == 1

    mat_xyz = Rotation.from_euler(seq='xyz', angles=theta, degrees=False).as_matrix()
    assert mat_xyz[0, 0] == 1

    print(f'{mat_zyx=} //\n {mat_xyz=}')





if __name__=='__main__':
    # test_rotation_decompose()
    # test_rotation_matrix()
    # test_rotation_matrix()
    test_rotation_decompose()