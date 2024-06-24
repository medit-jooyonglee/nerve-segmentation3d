import os
import json
import numpy as np
import glob
from trainer import image_utils
from typing import List, Dict, Tuple
# from commons import ge
from trainer import get_logger, diskmanager
from trainer.dataloder import ConfigDataset
from .augmentio import Augmentations


class SegmentData:
    def __init__(self):

        # source volume (d, h, w), normalize volume
        self.volume = np.ndarray([])
        # spacing (3,)
        self.spacing = np.ndarray([])
        # windowing min/max
        self.windowing = np.ndarray([])

        # segment volume (d0, h0, w0) d0 <= d && h0 <= h && w0 <- w
        self.segment = np.ndarray([])
        # the box of segment (6,) = (z0, y0, x0, z1, y1, x1)
        self.seg_bbox = np.ndarray([])

        self.filename = ''

    def fit_shape_source_and_segment(self):
        """
        source-volume & gt shape 이 다른 경우 roi정보를 참조하여 source을 cropping한다
        보통 source volume 크기 >= gt-volume

        """
        if self.volume.shape == self.segment.shape:
            pass
        else:
            volume_shape, seggment_shape = np.array(self.volume.shape), np.array(self.segment.shape)
            if np.all(volume_shape >= seggment_shape):
                i0, j0, k0, i1, j1, k1 = self.seg_bbox
                self.volume = self.volume[i0:i1, j0:j1, k0:k1].copy()
                assert self.volume.shape == self.segment.shape

    def parse(self):
        return self.volume, self.segment

    @property
    def nbytes(self):
        return SegmentData.bytes_nested(self.__dict__, 0)

    @staticmethod
    def bytes_nested(some_dict, total):
        if isinstance(some_dict, dict):
            for v in some_dict.values():
                if isinstance(v, np.ndarray):
                    total += v.nbytes
                elif isinstance(v, (list, dict)):
                    total = SegmentData.bytes_nested(v, total)

        elif isinstance(some_dict, list):
            for v in some_dict:
                if isinstance(v, np.ndarray):
                    total += v.nbytes
                elif isinstance(v, (list, dict)):
                    total = SegmentData.bytes_nested(v, total)
        return total



class SegmentDataset(ConfigDataset):
    def __init__(self, args, reuse_number=15, datatype='split', used_detection=False):
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
        super(SegmentDataset, self).__init__()
        assert datatype in ['compose', 'split']

        self.en_augmented = getattr(args, 'en_augmented', True)

        self.args = args

        self.edge_type = args.get('edge_type', 'pingpong')

        # source and gt
        self.source_path = []
        self.gt_path = []

        assert len(self.source_path) == len(self.gt_path), 'not same pair image'

        logger = get_runtime_logger()
        logger.info('total dataset:positive #{} '.format(len(self.source_path)))

        # (0, 1, 0, 1)
        upper = np.concatenate([np.arange(11, 19)[::-1], np.arange(21, 29)])
        lower = np.concatenate([np.arange(41, 49)[::-1], np.arange(31, 39)])


        self.fdi2label = np.arange(1000, dtype=np.uint8)
        # self.fdi2label[left_nerve] = 1
        # self.fdi2label[right_nerve] = 2



        self.datas: List[SegmentData] = []

        self.reuse_max_num = reuse_number
        self.reuse_count = -1
        self.data_index = -1

        self.datatype = datatype

        # 데이터 로드한 마지막 파일 번호
        self.last_loaded_file_index = 0

        self.max_bytes = 3 * (1024 ** 3)  # 10GB
        # self.max_bytes = 1  # 10GB
        # self.max_bytes = -1  # 10GB
        use_blurblock = args.get('use_blurblock')
        self.aug = Augmentations(use_blurblock)

        # mask dilation iteration
        self.mask_dilate_iter = 5

        self._reuse_max_num_prev = self.reuse_max_num
        self._is_train = True

    def __repr__(self):
        return 'pid:[{}]  {}:id[{}] : data #{} - loaded #{} // reuse count(cur/max) : {} / {} : {} MB'.format(
            os.getpid(), self.__class__.__name__, id(self), len(self.source_path), len(self.datas),
            self.reuse_count,
            self.reuse_max_num,
            self.nbytes / (1024 * 1024)
        )

    def eval(self):
        if self._is_train:
            self.en_augmented = False
            self._reuse_max_num_prev = self.reuse_max_num
            self.reuse_max_num = 1
            self._is_train = False
        else:
            self._is_train = True
            pass

    def train(self, state=True):
        if state:
            self.en_augmented = True
            self.reuse_max_num = self._reuse_max_num_prev
        else:
            self.eval()

    def prediction_collate(self, output):
        pass

    @staticmethod
    def search_file_pairs(path_pairs) -> List[Tuple[str, str]]:
        src_list = []
        gt_list = []
        for path_pair in path_pairs:
            src_path, tar_path = path_pair

            src_files = diskmanager.deep_serach_files(src_path, ['.npy'])
            gt_files = diskmanager.deep_serach_files(tar_path, ['.npy'])

            src_list += src_files
            gt_list += gt_files

        # src_path, tar_path = path_pair
        #
        # src_files = diskmanager.deep_serach_files(src_path, ['.npy'])
        # gt_files = diskmanager.deep_serach_files(tar_path, ['.npy'])

        assert len(src_list) == len(gt_list), 'not same pair image'
        return [(src, gt) for src, gt in zip(src_list, gt_list)]

    @classmethod
    def create_datasets(cls, dataset_config, phase, **kwargs):
        paths = dataset_config[phase].get('file_paths', ['', ''])[0]
        c = cls(phase)
        c.init_load(paths)

        return c

    def init_load(self, path_pair):
        assert all([os.path.exists(path) for path in path_pair])
        src, tar = path_pair
        self.data_clear()
        # assert os.path.exists(path), 'invlaid path'
        # diskmanager.deep_serach_files()
        src_files = diskmanager.deep_serach_files(src, ['.npy'])
        gt_files = diskmanager.deep_serach_files(tar, ['.npy'])
        assert len(src_files) == len(gt_files)
        self.source_path.extend(src_files)
        self.gt_path.extend(gt_files)
        return self

    def set_data_path_pair(self, file_pairs: List[Tuple[str]]):
        """
        list of 'source & gt file pairs'
        Parameters
        ----------
        file_pairs :

        Returns
        -------

        """
        if file_pairs:
            self.gt_path.clear()
            self.source_path.clear()
            assert len(file_pairs[0]) == 2
            srcs = [f[0] for f in file_pairs]
            tars = [f[1] for f in file_pairs]
            self.set_data_path(srcs, tars)
        return self

    def set_data_path(self, soure_files, target_files):
        self.gt_path.clear()
        self.source_path.clear()
        self.gt_path.extend(target_files)
        self.source_path.extend(soure_files)
        self.reuse_count = -1
        self.data_index = -1
        self.last_loaded_file_index = 0
        self.data_clear()

    def __len__(self):
        return int(len(self.source_path)) * self.reuse_max_num

    def __next__(self):
        return self[self.data_index + 1]

    def __getitem__(self, index):
        pass
        # post 처리는 안에서 처리하는게 맞는것 같다.
        return self.get_item_pair(index)

    @staticmethod
    def post_process(arr):
        a, b = arr.min(), arr.max()
        return (arr - a) / (b - a)

    @property
    def nbytes(self):
        """
        numpy-array 메모리 할당량을 가져온다.(bytes 단위)
        Returns
        -------

        """

        return np.sum([v.nbytes for v in self.datas], dtype=np.int64)

    def data_clear(self):
        self.datas.clear()

    def get_item_from_index(self, index):
        self.data_clear()
        self.last_loaded_file_index = index
        return next(self)

    def _load_data(self, index) -> SegmentData:
        logger = get_runtime_logger()
        source = np.load(self.source_path[index], allow_pickle=True).item()
        gtdata = np.load(self.gt_path[index], allow_pickle=True).item()
        logger.debug(f'loaded:{self.source_path[index]}')

        segdata = SegmentData()

        # check key
        assert (('data' in source or 'volume' in source) and \
                'spacing' in source), 'not find key'
        volume = source.get('data') or source.get('volume')
        segdata.spacing = source.get('spacing')
        segdata.windowing = source.get('windowing')
        dmin, dmax = segdata.windowing
        segdata.volume = (volume.astype(np.float32) - dmin) / (dmax - dmin)

        seg_label = self.fdi2label[gtdata.get('data')]
        segdata.segment = seg_label.astype(np.float32)
        segdata.seg_bbox = gtdata.get('bbox')

        segdata.fit_shape_source_and_segment()
        return segdata

    def load_next_file_if_full_reused_or_empty(self):
        """
        1. reuse 최대치 일때 다음 데이터 읽거나
        2. 데이터 비어 있을 때(초기상태) 데이터를 읽어온다
        Returns
        -------
            bool:
                데이터를 읽어온 경우 True
                pass 상태일 때는 False

        """
        if (self.reuse_count + 1) == self.reuse_max_num or len(self.datas) == 0:
            self.reuse_count = -1
            self.data_index = -1
        else:
            return False

        self.data_clear()

        start = self.last_loaded_file_index

        total_bytes = 1000
        logger = get_runtime_logger()
        logger.debug('loading...')

        for _ in range(len(self.source_path)):
            try:
                segdata = self._load_data(start)
            except Exception as e:
                logger.error(e.args[0])
                continue
            self.datas.append(segdata)
            start = (start + 1) % len(self.source_path)

            if self.nbytes > self.max_bytes:
                self.last_loaded_file_index = start
                break
        return True

    # def fdi2label(self):

    def is_active_preprocess_mask(self):
        return hasattr(self.args, 'preprocess') and \
            self.args.__dict__.get('preprocess', {}).get('active', False)

    def preprocess_mask(self, mask):
        """
        preprocess 활성화 된 경우 전처리한다.
        그렇지 않는 경우는 입력데이터 그대로 반환한다.
        Parameters
        ----------
        mask : np.ndarray (d, h, w) 정수형 볼륨

        Returns
        -------
        np.ndarray 전처리된 mask

        """
        return mask

    def load_next_data(self):
        """
        학습에 필요한 (input, target) 데이터를 가져온다.
        Returns
        -------
            tuple[np.ndarray, np.ndarray]
            inputs & target pair

        """

        self.load_next_file_if_full_reused_or_empty()

        # increase data number
        next_index = self.data_index + 1
        if next_index == len(self.datas):
            self.reuse_count += 1
            next_loaded = self.load_next_file_if_full_reused_or_empty()
            next_index = 0 if next_loaded else next_index

        self.data_index = next_index % len(self.datas)
        #
        #     self.reuse_count += 1
        target_shape = [128, ] * 3
        i = self.data_index

        # self.datas[]
        # bbox = self.roi_bboxes[i]
        src, tar = self.datas[i].parse()
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
            src_aug, tar_aug = [np.squeeze(a) for a in augs]
        else:
            src_aug, tar_aug = src_crop, tar_crop
        return src_aug, tar_aug

    # @property
    def gen_augment_param(self):
        augment_param = {
            "rotate": np.random.uniform(-np.pi / 8, np.pi / 8, [3]),
            "scale": np.random.uniform(0.85, 1.2, [3]),
            "translate": np.random.uniform(-0.15, 0.15, [3])
        }
        return augment_param

    def get_item_pair(self, index):
        def model_full_segment(_model, *args):
            res = _model.full_segment(*args)
            return res

        return self.load_next_data()



class MultiSegmentDataset(SegmentDataset):
    # # roi detection 을 위한 모델
    roi_detection_model = None

    def __init__(self, args, reuse_number=15, datatype='split', used_detection=False):
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
        super(MultiSegmentDataset, self).__init__(args, reuse_number=reuse_number, datatype=datatype,
                                                  used_detection=used_detection)

        # import tools.reluapploader.reader as relureader
        # relutable = relureader.read_relauapp_table('relutable.yaml')

        # re-define fdi2label for converting mask
        # left_nerve, right_nerve = relutable['Left_mandibular_canal'], relutable['Right_mandibular_canal']
        self.fdi2label = np.zeros([50], dtype=np.uint8)
        num_class_tooth = 2
        class_count = 0
        self.label2text: dict[int, str] = dict()
        relutable = dict()
        # 치아 말고 나머지는 각각의 클래스로
        for key, val in relutable.items():
            if key.find('Tooth') >= 0:
                print(key, 'tooth')
                number = int(key.split('_')[-1])

                if num_class_tooth == 2:
                    # 상악은 1 하악은 2
                    if number < 30:
                        label = 1
                    else:
                        label = 2
                    self.fdi2label[val] = label
                    self.label2text[label] = key
                else:
                    raise ValueError
            else:
                print(key)
                label = (num_class_tooth + 1) + class_count
                self.fdi2label[val] = label
                self.label2text[label] = key

                class_count += 1

        # print(self.label2text)

    @staticmethod
    def search_file_pairs(path_pair) -> List[Tuple[str, str]]:
        return SegmentDataset.search_file_pairs(path_pair)

    # def get_last_filename(self):
    #     if self.data_index < len(self.source_path):
    #         return self.source_path[self.data_index]
    #     else:
    #         return ''
    def get_last_filename(self) -> str:
        if self.data_index < len(self.datas):
            return self.datas[self.data_index].filename
        else:
            return ''

    def _load_data(self, index) -> SegmentData:
        logger = get_runtime_logger()

        source = np.load(self.source_path[index], allow_pickle=True).item()
        gtdata = np.load(self.gt_path[index], allow_pickle=True).item()
        logger.debug(f'loaded:{self.source_path[index]}')

        segdata = MultiSegmentData()
        # check key
        assert (('data' in source or 'volume' in source) and \
                'spacing' in source and \
                ('windowing' in source or 'range' in source)), 'not find key'
        volume = source.get('data') if 'data' in source else source.get('volume')
        segdata.spacing = source.get('spacing')
        segdata.windowing = source.get('windowing') if 'windowing' in source else source.get('range')
        dmin, dmax = segdata.windowing

        assert (('mask' in gtdata) and \
                'bbox' in gtdata), 'not find key'

        segdata.volume = np.clip((volume.astype(np.float32) - dmin) / (dmax - dmin), 0, 1)

        seg_label = self.fdi2label[gtdata.get('mask')]
        segdata.segment = seg_label.astype(np.float32)
        segdata.filename = self.source_path[index]
        # segdata.seg_box_per_label = gtdata.get('bbox')
        bboxes = {self.fdi2label[lab]: box for lab, box in gtdata.get('bbox').items() if self.fdi2label[lab] > 0}
        segdata.seg_box_per_label = bboxes

        segdata.fit_shape_source_and_segment()
        return segdata

    def load_next_data(self):
        """
        학습에 필요한 (input, target) 데이터를 가져온다.
        Returns
        -------
            tuple[np.ndarray, np.ndarray]
            inputs & target pair

        """

        self.load_next_file_if_full_reused_or_empty()

        # increase data number
        next_index = self.data_index + 1
        if next_index == len(self.datas):
            self.reuse_count += 1
            next_loaded = self.load_next_file_if_full_reused_or_empty()
            next_index = 0 if next_loaded else next_index

        self.data_index = next_index % len(self.datas)
        #
        #     self.reuse_count += 1
        target_shape = [128, ] * 3
        i = self.data_index

        # self.datas[]
        # bbox = self.roi_bboxes[i]
        src, tar = self.datas[i].parse()
        # src, tar = self.datas[i], self.gt_masks[i]
        bbox = np.concatenate([np.zeros_like(src.shape), src.shape])
        aug_param = self.gen_augment_param() if self.en_augmented else {}
        src_crop = image_utils.auto_cropping_keep_ratio_with_box_augment(src, bbox, target_shape,
                                                                         augent_param=aug_param)
        tar_crop = image_utils.auto_cropping_keep_ratio_with_box_augment(tar, bbox, target_shape,
                                                                         augent_param=aug_param, method='nearest')

        # 소프트티슈 특정 데이터 없는거가 존재해서 mask로 처리해서 학습에 기여하지 않도록 weight값 반환
        soft_tissue_label = 40
        if soft_tissue_label in self.datas[i].seg_box_per_label:
            weight = np.ones_like(tar_crop)
            # weight = tar_crop > 0
        else:
            weight = tar_crop != self.fdi2label[soft_tissue_label]
        # tar_crop

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
        return src_aug, tar_aug, weight



class MultiSegmentRoIDataset(MultiSegmentDataset):
    # # roi detection 을 위한 모델
    roi_detection_model = None

    def __init__(self, args, reuse_number=15, datatype='split', used_detection=False):
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
        super(MultiSegmentRoIDataset, self).__init__(args, reuse_number=reuse_number, datatype=datatype,
                                                     used_detection=used_detection)
        self.fdi2label = np.arange(1000, dtype=np.uint8)


    def load_next_data(self):
        """
        학습에 필요한 (input, target) 데이터를 가져온다.
        Returns
        -------
            tuple[np.ndarray, np.ndarray]
            inputs & target pair

        """

        self.load_next_file_if_full_reused_or_empty()

        # increase data number
        next_index = self.data_index + 1
        if next_index == len(self.datas):
            self.reuse_count += 1
            next_loaded = self.load_next_file_if_full_reused_or_empty()
            next_index = 0 if next_loaded else next_index

        self.data_index = next_index % len(self.datas)
        #
        #     self.reuse_count += 1
        target_shape = [128, ] * 3
        i = self.data_index

        # self.datas[]
        # bbox = self.roi_bboxes[i]
        src, tar = self.datas[i].parse()
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
