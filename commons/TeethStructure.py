import numpy as np
#
class StatModelBase(object):
    def __init__(self, tau=None, pose=np.zeros([3]), scale=np.ones([3]), euler=np.zeros([3]), mat4x4=np.eye(4)):
        self.pose = pose
        self.scale = scale
        self.euler_theta = euler
        self.matrix4x4 = mat4x4
        self.tau = tau

        self.pose_components_ = np.eye(3)
        self.euler_components_ = np.eye(3)
        self.scale_components_ = np.eye(3)


class TeethStructure(object):
    def __init__(self, voxel, spacing, taus):
        # voxel statistical model from mesh, refer to method, statmodel.py method "converting_voxel" line 603~
        self.voxel = voxel
        self.spacing = spacing
        self.taus = taus

        # dictionary of "StatModelBase", StatModelBase contain stdev&mean of 'pose, orient, scale'
        self.statmodel = dict()

        # staatistical pose orthonal basis (in case(person))
        self.teeth_pose = dict()

    def get_transform_dict(self):
        transform_dict = {}
        for tau, value in self.statmodel.items():
            if isinstance(value, StatModelBase):
                transform_dict[tau] = value.matrix4x4
        return transform_dict

    def update_stat(self,  stat_items):
        """
        :param stat_items: instance of StatModelBase
        :return:
        """
        for tau, values in stat_items.items():
            it = StatModelBase()
            it.__dict__.update(values.__dict__)
            self.statmodel[tau] = it
