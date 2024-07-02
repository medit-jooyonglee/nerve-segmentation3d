from torch.utils.data import Dataset
import json
import os.path
import numpy as np
import glob
from trainer import get_logger
from tools import vtk_utils, image_utils
from dataset.mha import read_image_volume
from dataset.augmentio import Augmentations
import logging

logger = get_logger('miccaitoothfairy')
# from .nervedataset import NerveBoxDataset, NerveDetectionSet
LEFT = 1
RIGHT = 2

def check_all_data(src_files, label_files):
    def basename(filename): return os.path.splitext(os.path.basename(filename))[0]
    def split_id(filename, i): return int(filename.split('_')[-i])

    src_ids = [split_id(basename(name), 2) for name in src_files]
    label_ids = [split_id(basename(name), 1) for name in label_files]
    np.testing.assert_array_equal(src_ids, label_ids)



def sort_file(src_files, label_files):
    
    return sorted(src_files), sorted(label_files)
    #    def basename(filename): return os.path.splitext(os.path.basename(filename))[0]
#    def split_id(filename, i):i
#        splits = filename.split('_')[-i]
#        return int(splits), len(splits)
#
#
#    src_ids = [split_id(basename(name), 2) for name in src_files]
#    label_ids = [split_id(basename(name), 1) for name in label_files]
#    np.testing.assert_array_equal(src_ids, label_ids)


def data_split_path(src_pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/imagesTr',
                    label_pathname='D:/dataset/miccai/Dataset112_ToothFairy2/labelsTr', train_weights=0.9,
                    save_path='./'):

    src_files = glob.glob(os.path.join(src_pathname, '*.mha'))
    label_files = glob.glob(os.path.join(label_pathname, '*.mha'))
    print(f'file size {len(src_files)}')
#    print(src_files)

    assert len(src_files) == len(label_files)
    try:
        check_all_data(src_files, label_files)
    except Exception as e:
        src_files, label_files = sort_file(src_files, label_files)
        check_all_data(src_files, label_files)
    num_split = int(train_weights * len(src_files))
    num_valid = len(src_files) - num_split
    train_srcs, train_labels = src_files[:num_split], label_files[:num_split]
    valid_src, valid_labels = src_files[-num_valid:], label_files[-num_valid:]
    print(f'split train/valid {len(train_srcs)} / {len(valid_src)} ')
    datas = {
        'train': {
            'source': train_srcs,
            'label': train_labels,
        },
        'valid': {
            'source': valid_src,
            'label': valid_labels,
        },
    }
    print(f'save complete{save_path} ')
    with open(os.path.join(save_path, 'data_split_miccai.json'), 'w') as f:
        json.dump(datas, f, indent='\t')

def data_split_create_list():
#    src_pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/imagesTr'
#    label_pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/labelsTr'

    src_pathname = '/data1/miccai/Dataset112_ToothFairy2/imagesTr'
    label_pathname = '/data1/miccai/Dataset112_ToothFairy2/labelsTr'


#    label_pathname = 'D:/dataset/miccai/Dataset112_ToothFairy2/labelsTr'
    src_files = glob.glob(os.path.join(src_pathname, '*.mha'))
    label_files = glob.glob(os.path.join(label_pathname, '*.mha'))
    print(len(src_files), len(label_files))

    assert len(src_files) == len(label_files)
    print(src_files[:5])
    print(label_files[:5])

    save_path = 'd:/temp/tooth_fairy_cpature'
    os.makedirs(save_path, exist_ok=True)
    for src, gt in zip(src_files, label_files):
        src_image, src_ref = read_image_volume(src)
        gt_image, label_ref = read_image_volume(gt, normalize=False)

        # print(src_volume.shape, label_volume.shape)
        src_clip_image = src_image[-gt_image.shape[0]:]
        save_name = os.path.join(save_path, os.path.splitext(os.path.basename(src))[0] + '.png')
        vtk_utils.split_show([
            src_clip_image
        ], [
            gt_image
        ],
            [src_clip_image, *vtk_utils.auto_refinement_mask(gt_image)]
        ,
        show=False, image_save=True, savename=save_name, cam_direction=(3, 3, 2))


class NerveMICCAISet(Dataset):
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
        # use_blurblock = kwargs.get('use_blurblock')
        # self.aug = Augmentations(use_blurblock)
        super(NerveMICCAISet, self).__init__(**{})

        file_infos = kwargs.get('paths', [('', '')])

        self.sources = []
        self.labels = []
        for jsoname, theme in file_infos:
            assert os.path.exists(jsoname), f'cannot find filenamae:{jsoname}'
            with open(jsoname, 'r') as f:
                data = json.load(f)

            data_pair = data.get(theme, {})
            sources, labels = data_pair.get('source', []), data_pair.get('label', [])
            assert len(sources) == len(labels), 'not same the source & labels'
            self.sources.extend(sources)
            self.labels.extend(labels)

        if len(self.sources) == 0:
            logger.error('cannot find the source label info. call method "data_split_path(...)"')

        self.name = kwargs.get('name', '')
        self._num_reuse_image = 10

        self._num_save_image = 20
        self._data = dict()
        self.en_augmented = True
        self.aug = Augmentations(False)

        # self._temp_data =
    def __len__(self):
        return len(self.sources) * self._num_reuse_image

    def index2real(self, val):
        # 0, 0, 0..., 0), ()1, 1, 1, 1...1)
        return val // self._num_reuse_image

    def real2index(self, val):
        return self._num_reuse_image * val

    def clear_items(self):
        if len(self._data) > self._num_save_image:
            logger.debug(f'clear items the data {len(self._data)}')

            self._data.clear()

    def _load_data(self, index):
        # if len(self.sources)  0:
        real_index = self.index2real(index)

        self.clear_items()

        if not real_index in self._data:
            src, gt = self.sources[real_index], self.labels[real_index]
            src_image, src_ref = read_image_volume(src)
            gt_image, label_ref = read_image_volume(gt, normalize=False)
            mapper = np.zeros([100], dtype=np.uint8)
            # (3, 4) ->(1, 2)
            mapper[np.array([3, 4])] = np.array([1, 2])
            gt_image = mapper[gt_image]
            # source & label 크기가 맞지 않아서 크기 보정
            gt_image_exp = np.zeros_like(src_image, dtype=gt_image.dtype)
            gt_image_exp[-gt_image.shape[0]:] = gt_image

            coords = [np.argwhere(gt_image_exp==v) for v in [1, 2]]
            items = (src_image.astype(np.float32), gt_image_exp, coords)
            self._data[real_index] = items
        else:
            logger.debug(f'reuse items {real_index} / {index}')
            items = self._data[real_index]
        return items

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
        item = self._load_data(index)
        # if self.always_file_load:

        src, tar, coords = item
        target_shape = [128, ] * 3
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



class NerveMICCAIBoxSet(NerveMICCAISet):
    # # roi detection 을 위한 모델
    roi_detection_model = None

    def __init__(self, **kwargs):

        # use_blurblock = kwargs.get('use_blurblock')
        # self.aug = Augmentations(use_blurblock)
        super(NerveMICCAIBoxSet, self).__init__(**kwargs)
        #
        # file_infos = kwargs.get('paths', [('', '')])
        #
        # self.sources = []
        # self.labels = []
        # for jsoname, theme in file_infos:
        #     assert os.path.exists(jsoname), f'cannot find filenamae:{jsoname}'
        #     with open(jsoname, 'r') as f:
        #         data = json.load(f)
        #
        #     data_pair = data.get(theme, {})
        #     sources, labels = data_pair.get('source', []), data_pair.get('label', [])
        #     assert len(sources) == len(labels), 'not same the source & labels'
        #     self.sources.extend(sources)
        #     self.labels.extend(labels)
        #
        # if len(self.sources) == 0:
        #     logger.error('cannot find the source label info. call method "data_split_path(...)"')
        #
        # self.name = kwargs.get('name', '')
        # self._num_reuse_image = 10
        #
        # self._num_save_image = 20
        # self._data = dict()
        # self.en_augmented = True
        # self.aug = Augmentations(False)

    #     # self._temp_data =
    # def __len__(self):
    #     return len(self.sources) * self._num_reuse_image
    #
    def get_obb_augment_param(self):
        # if self.en_augmented:
        return {
            "theta": np.random.uniform(-np.pi / 18, np.pi / 18, [3]),
            "scale": np.random.uniform(0.9, 1.1),
            "translate": np.random.uniform(-2, 2, [3])
        }


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

        item = self._load_data(index)

        src_volume, target_volume, sample_coords = item

        volumes_list = [
            src_volume,
            target_volume,
        ]

        pool_seed = np.array([64, 32, 32])

            # coords
        scale = np.random.uniform(0.9, 1.1)

        # for coords in sample_coords:
        coords = sample_coords[np.random.choice(len(sample_coords))]
        coords = coords if coords.shape[0] > 3 else np.random.uniform(30, 60, [10, 3])
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

        return src_aug, tar_aug

if __name__ == '__main__':
    # main()
    # data_split_create_list()
    # glob.glob()
    # filename = ''
    # data_split_path()

    
    src_pathname = '/data1/miccai/Dataset112_ToothFairy2/imagesTr'
    label_pathname = '/data1/miccai/Dataset112_ToothFairy2/labelsTr'
    data_split_path(src_pathname, label_pathname, save_path='./dataset')
    #    data_split_create_list()
