import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from tools import image_utils
from trainer import diskmanager
from .augmentio import Augmentations
from trainer.utils import get_logger
from tools.geometry_numpy import norm_boxes

get_runtime_logger = get_logger()

LEFT = 1
RIGHT = 2

class EmtpyDataset(Dataset):
    def __init__(self, **kwargs):
        super(EmtpyDataset, self).__init__()
        pass
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x = np.random.randn(128, 128, 128)
        return x[None], x.astype(np.int64).clip(0, 1)

class EmtpyROIDataset(Dataset):
    def __init__(self, **kwargs):
        super(EmtpyROIDataset, self).__init__()
        pass
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x = np.random.randn( 128, 64, 32)
        return x[None], x.astype(np.int64).clip(0, 1)


class NerveDetectionSet(Dataset):
    # # roi detection 을 위한 모델
    roi_detection_model = None

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        args : object
            기타 파라미터
        reuse_count : int
            학습 데이터 재활용 개수. 디폴트=1, 1epoch data size = reuse_count x actual_data_size
        datatype : str
            학습 데이터 parsing 옵션(학습 데이터 저장 옵션( see: prepare_dataset.py))
            'compose' : teeth roi croppaed & input-volume + mask
            'split' : 플래닝 아이템 개별로 관리
        used_detection: bool
            detection module 사용할 지 옵션
        """
        super(NerveDetectionSet, self).__init__()
        use_blurblock = kwargs.get('use_blurblock')
        self.aug = Augmentations(use_blurblock)

        self.name = kwargs.get('name', '')
        self.source = []
        self.labels = []
        path_pait_list = kwargs.get('paths', [('', '')])


        for items in path_pait_list:
            source_path, label_path = items
            src_files = diskmanager.deep_serach_files(source_path, ['.npy'])
            gt_files = diskmanager.deep_serach_files(label_path, ['.npy'])
            self.source.extend(src_files)
            self.labels.extend(gt_files)
        self.items = {}
        self.en_augmented = True# self
        self.always_file_load = False
        self._index = 0

    # def splits(self):


    def __len__(self):
        return len(self.source)

    def _load_data(self, index):
        def normalize_volume(volume):
            dtypeinfo = np.iinfo(volume.dtype)
            return (volume - dtypeinfo.min) / (dtypeinfo.max - dtypeinfo.min)

        def recon_volume2(coords, shape, coords_order='xyz', dtype='float32'):
            lefts, rights = coords
            # target_shape = shape if target_shape is None else np.asarray(target_shape)
            volume = np.zeros([*shape], dtype=dtype)
            for i, (coord, label) in enumerate(zip([lefts, rights], [LEFT, RIGHT])):
                i0, j0, k0 = [np.squeeze(v).astype(np.int32) for v in np.split(coord, 3, axis=-1)]
                if coords_order == 'xyz':
                    volume[k0, j0, i0] = label
                else:
                    volume[i0, j0, k0] = label

            return volume
        index = index % len(self.source)
        src, gt = self.source[index], self.labels[index]
        src_item = np.load(src, allow_pickle=True).item()
        gt_item = np.load(gt, allow_pickle=True).item()

        norm_volume = normalize_volume(src_item.get('data'))

        coord_order = gt_item['order']
        # gt_item['data']
        # gt_item
        coords = gt_item['data']

        table = gt_item['table']
        left_right_key = [k for p in ['left', 'right'] for k, v in table.items() if p == v]
        left_right_coords = [coords[key] for key in left_right_key]
        # [k for k, v in gt_item['table'].items() if v == 'left']
        label_volume = recon_volume2(left_right_coords, norm_volume.shape, coords_order=coord_order, dtype='uint8')

        return norm_volume, label_volume, left_right_coords

    # @property
    def gen_augment_param(self):
        augment_param = {
            "rotate": np.random.uniform(-np.pi / 8, np.pi / 8, [3]),
            "scale": np.random.uniform(0.85, 1.2, [3]),
            "translate": np.random.uniform(-0.15, 0.15, [3])
        }
        return augment_param

    def __getitem__(self, index):
        # self._index = 0
        # sself.so
        # np.
        if self.always_file_load:
            item = self._load_data(index)
        else:
            if not index in self.items:
                item = self._load_data(index)
                self.items[index] = item
            else:
                item = self.items[index]
        src, tar, coords = item
        target_shape = [128,] * 3
        # src, tar = self.datas[i], self.gt_masks[i]
        bbox = np.concatenate([np.zeros_like(src.shape), src.shape])
        aug_param = self.gen_augment_param() if self.en_augmented else {}
        src_crop = image_utils.auto_cropping_keep_ratio_with_box_augment(src, bbox, target_shape,
                                                                         augent_param=aug_param)
        tar_crop = image_utils.auto_cropping_keep_ratio_with_box_augment(tar, bbox, target_shape,
                                                                         augent_param=aug_param, method='nearest')
        if self.en_augmented:
            # (d, h, w)->(1, d, h, w)->(d, h, w),src 데이터 intensity augmentation 적용하지 않도록 처리, spatial aug만 적용
            augs = self.aug.apply_transform([src_crop[np.newaxis], tar_crop[np.newaxis]], mode='all',
                                            stat=['scalar', 'label'])
            src_aug, tar_aug = augs
            tar_aug = np.squeeze(tar_aug)
        else:
            src_aug, tar_aug = src_crop, tar_crop
            src_aug = src_aug[None]
        tar_aug = tar_aug.astype(np.int64)

        return src_aug, tar_aug


    def __next__(self):
        self._index += 1
        return self[self._index]


class NerveBoxDataset(NerveDetectionSet):
    def __init__(self, **kwargs):
        super(NerveBoxDataset, self).__init__(**kwargs)

        self.datas = self.items
        self.detection = None
        self.used_detection = False

    def get_obb_augment_param(self):
        # if self.en_augmented:
        return {
            "theta": np.random.uniform(-np.pi / 18, np.pi / 18, [3]),
            "scale": np.random.uniform(0.9, 1.1),
            "translate": np.random.uniform(-2, 2, [3])
        }

    def __len__(self):
        # 좌, 우 두개 들어 있으므로
        # 한번에 하나씪 샘플링
        return len(self.source) * 2

    def __getitem__(self, item):

    # def load_next_data(self):
        """
        학습에 필요한 (input, target) 데이터를 가져온다.
        Returns
        -------
            tuple[np.ndarray, np.ndarray]
            inputs & target pair

        """
        index = item
        # np.
        if self.always_file_load:
            item = self._load_data(index)
        else:
            if not index in self.items:
                item = self._load_data(index)
                self.items[index] = item
            else:
                item = self.items[index]
        src_volume, target_volume, coords = item

        if self.used_detection and self.detection is not None:
            pred_segment = self.detection(src_volume)
            sample_coords = [np.argwhere(pred_segment == i) for i in [LEFT, RIGHT]]
            # sample_coords =
        else:
            # pred_segment = target_volume
            sample_coords = coords
            # sample_coords
        volumes_list = [
            src_volume,
            target_volume,
        ]
        # except background class

        pool_seed = np.array([64, 32, 32])

            # coords
        scale = np.random.uniform(0.9, 1.1)

        # for coords in sample_coords:
        coords = sample_coords[np.random.choice(len(sample_coords))]
        augment_param = self.get_obb_augment_param() if self.en_augmented else {}
        sample_volumes, obb_meta, warp_points = image_utils.volume_sampling_from_coords(volumes_list,
                                                                                        coords, pool_seed,
                                                                                        scale, augment_param,
                                                                                        extend_size=1.10,
                                                                                        return_warp_points=True)


        volumes, mask = sample_volumes

        mask = (mask > 0).astype(np.float32)

        if self.en_augmented:
            # (d, h, w)->(1, d, h, w)->(d, h, w),src 데이터 intensity augmentation 적용하지 않도록 처리, spatial aug만 적용
            augs = self.aug.apply_transform([volumes[np.newaxis], mask[np.newaxis]], mode='all',
                                            stat=['scalar', 'label'])
            src_aug, tar_aug = augs
            tar_aug = np.squeeze(tar_aug)
        else:
            src_aug, tar_aug = volumes, mask
            src_aug = src_aug[None]

        tar_aug = tar_aug.astype(np.int64)
        # volumes = volumes[np.newaxis]
        # volumes = np.stack(volumes, axis=0)
        # masks = np.stack(masks, axis=0)
        # vol_stack = vol[np.neaaxis]
        return src_aug, tar_aug


class EmptyDataTeethSet(Dataset):
    def __init__(self, **kwargs):
        super(EmptyDataTeethSet, self).__init__()
        pass
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        segment_ch = 3
        coord_ndim = 3
        # x = np.ran
        # dom.randn(1, 128, 128, 128)
        shape = (128, ) * 3
        x = np.random.randn(1, *shape)
        seg = np.random.randn(2, *shape).clip(0, segment_ch - 1).astype(np.int64)
        offset = np.random.randn(coord_ndim, *shape)
        return x, ((seg, offset), )
        # return x[None], x.astype(np.int64).clip(0, 1)


class EmptyROISegDataTeethSet(Dataset):
    def __init__(self, **kwargs):
        super(EmptyROISegDataTeethSet, self).__init__()

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        segment_ch = 3
        coord_ndim = 3
        # x = np.ran
        # dom.randn(1, 128, 128, 128)
        shape = (128, ) * 3
        x = np.random.randn(1, *shape)
        seg = np.random.randn(2, *shape).clip(0, segment_ch - 1).astype(np.int64)
        offset = np.random.randn(coord_ndim, *shape)
        return x, ((seg, offset), )
        # return x[None], x.astype(np.int64).clip(0, 1)