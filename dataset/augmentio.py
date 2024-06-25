import torch
import numpy as np
import torchio as tio
from torchio import Subject
from collections import OrderedDict
from skimage import morphology
from scipy.ndimage import gaussian_filter

from typing import List

from commons import timefn2


def load_torch_dataset():
    colin = tio.datasets.Colin27()
    transforms_dict = {
        tio.RandomAffine(): 0.75,
        tio.RandomElasticDeformation(): 0.25,
    }  # Using 3 and 1 as probabilities would have the same effect

    colin.applied_transforms
    transform = tio.OneOf(transforms_dict)

    img = colin.get_first_image()
    import numpy as np
    from tools import vtk_utils
    i = np.array(img)

    scale = img['data'].max() - img['data'].min()
    norm_img = img['data'] / scale
    vtk_utils.show_actors([norm_img.numpy()])

    vtk_utils.show_actors([i])

    transformed = transform(colin)

    random_affine = tio.RandomAffine()
    res = random_affine.apply_transform(colin)

    for k in colin.keys():
        vtk_utils.show_actors([
            np.squeeze(np.array(colin[k]))
        ])

    img = colin.get_first_image()

    def to_normalize_numpy(tio_img):
        if tio_img.data.is_floating_point():
            fmin, fmax = tio_img.data.min(), tio_img.data.max()
            return np.squeeze(((tio_img.data - fmin) / (fmax - fmin)).numpy())
        else:
            return np.squeeze(tio_img.data.numpy())

    # img.data
    vtk_utils.show_actors([
        # to_normalize_numpy(img)
        *[to_normalize_numpy(v) for v in res.values()]
    ])

    # tio.LabelMap

    import torch

    dmri = tio.ScalarImage(tensor=torch.rand(32, 128, 128, 88))
    label_img = tio.LabelMap(tensor=(dmri.data > .5).to(torch.int64))

    img_set = tio.Subject({
        'img': dmri,
        'label': label_img
    })
    # res2 = random_affine.apply_transform({'a': dmri, 'b': label_img})
    res2 = random_affine.apply_transform(img_set)


@timefn2
def rotation(src: torch.Tensor, tar: torch.Tensor):
    img_set = tio.Subject({
        'image': tio.ScalarImage(tensor=src),
        'label': tio.LabelMap(tensor=tar)
    })

    random_affine = tio.RandomAffine(degrees=50)
    res = random_affine.apply_transform(img_set)
    return res['image'].data, res['label'].data


def random_ghostring(src: torch.Tensor, tar: torch.Tensor):
    img_set = tio.Subject({
        'image': tio.ScalarImage(tensor=src),
        'label': tio.LabelMap(tensor=tar)
    })
    random_ghost = tio.RandomGhosting()
    res = random_ghost.apply_transform(img_set)
    return res['image'].data, res['label'].data


class BlurBlock(tio.SpatialTransform):
    def __init__(self, block_size=15):
        super().__init__()
        self.block_size = block_size

    def apply_transform(self, subject: Subject) -> Subject:
        subject = self._blur_block(subject)
        return subject

    def _blur_block(self, subject):
        """
        Augmentation 데이터 생성을 위한 weighted mask 생성
        (신경관 일부분을 블러링 적용)

        Parameters
        ----------
        subject: torchio subject (scalar image, label map)
        coord: 신경관 일부분의 중앙 좌표점
        block_size: 신경관 일부분 크기
        alpha: 신경관 gt와 weighted mask 비율 (0~1)

        Returns
        -------
        subject: torchio subject (scalar image, weighted label map)
        """
        volumes, nerve = self.get_images(subject)
        volume = volumes.numpy()[0, :]
        nerve = np.squeeze(nerve.numpy())

        # randomly set parameter
        # sigma: 가우시안 블러링 파라미터 값
        # blcok_size
        beta = np.random.randint(5, 15)
        sigma = np.random.rand(1)[0] + 1
        block_size = np.random.randint(10, 40)

        # Generate random block mask
        diff = np.ceil(block_size / 2)
        random_mask = np.zeros(nerve.shape)

        coords = np.array(np.where(nerve == 1)).T
        if not coords.size > 0:
            return subject

        idx = np.random.randint(coords.shape[0])
        coord = coords[idx, :]

        block_coords = []
        for i, v in enumerate(coord):
            if v + diff > nerve.shape[i]:
                v_min = np.min([nerve.shape[i] - 1, v + diff]) - block_size
            elif v - diff < 0:
                v_min = np.max([0, v - diff])
            else:
                v_min = v - diff
            v_max = v_min + block_size
            block_coords.append([int(v_min), int(v_max)])

        random_mask[block_coords[0][0]:block_coords[0][1], block_coords[1][0]:block_coords[1][1],
        block_coords[2][0]:block_coords[2][1]] = 1
        weighted_mask = random_mask * nerve

        # Dilate mask
        weighted_mask = morphology.dilation(weighted_mask, morphology.ball(beta))

        # Blurring
        blurred_mask = gaussian_filter(weighted_mask, sigma)
        blurred_volume = gaussian_filter(volume, sigma * 3)

        # Generate Weighted mask
        tar = volume * (1 - blurred_mask) + blurred_volume * blurred_mask

        # return torchio subject with weighted label map
        tar = np.expand_dims(tar, axis=0)
        src2 = np.expand_dims(volumes.numpy()[1, :], axis=0)
        weighted_image = np.concatenate((tar, src2), axis=0)
        weighted_image = torch.as_tensor(weighted_image)
        subject['0'].set_data(weighted_image)

        return subject


class Augmentations:
    def __init__(self, custom_aug_nerve_seg=False):
        # https://torchio.readthedocs.io/transforms/augmentation.html
        self._spatial_augments = [
            # spatial
            (1.0, tio.RandomAffine()),
            (1.0, tio.RandomElasticDeformation(max_displacement=4.5)),
            # (4.0, tio.RandomFlip(axes=(1, 2), flip_probability=0.85)),
            (1.0, tio.RandomAnisotropy(downsampling=(1.5, 2.5))),
        ]

        self._intensity_augments = [
            # intensity
            (1.0, tio.RandomGhosting(intensity=(30 / 255, 100 / 255))),
            (1.0, tio.RandomMotion()),
            (1.0, tio.RandomSpike(intensity=(1 / 255, 3 / 255))),
            (1.0, tio.RandomBiasField()),
            (1.0, tio.RandomSpike(intensity=(2 / 255, 5 / 255))),
            (1.0, tio.RandomNoise(std=(0, 5 / 255))),
            (1.0, tio.RandomSwap(patch_size=10)),
        ]

        if custom_aug_nerve_seg:
            self._intensity_augments.append((1.0, BlurBlock()))

        self._all_augment_pair = [
            *self._spatial_augments,
            *self._intensity_augments
        ]

    def random_compose(self, augment_pairs=[]) -> List[tio.Transform]:
        augment_pairs = augment_pairs or self._all_augment_pair
        augments_list = [b for (a, b) in augment_pairs]
        probs = [a for (a, b) in augment_pairs]
        probs = np.array(probs) / np.sum(probs)

        num_comp = round(np.random.uniform(1, len(augments_list) * .3))
        num_comp = np.minimum(num_comp, len(augments_list))
        augs = np.random.choice(augments_list, size=num_comp, replace=False, p=probs)
        return augs.tolist()

    def apply_transform(self, image_list: List[np.ndarray], mode: str = 'all', stat=[]):
        assert mode in ['all', 'spatial', 'intensity']
        augs = []
        if mode == 'all':
            augs = self._all_augment_pair
        elif mode == 'spatial':
            augs = self._spatial_augments
        elif mode == 'intensity':
            augs = self._intensity_augments
        else:
            pass
        assert all([s in ('scalar', 'label') for s in stat])
        compose = self.random_compose(augs)
        # print(compose)
        compose_transform = tio.Compose(compose)
        images = OrderedDict()

        tioImage = {
            'scalar': tio.ScalarImage,
            'label': tio.LabelMap
        }

        for i, img in enumerate(image_list):
            if len(stat) == len(image_list):
                img2 = tioImage[stat[i]](tensor=img)
            else:
                if np.issubdtype(img.dtype, np.integer):
                    img2 = tio.LabelMap(tensor=img)
                else:
                    img2 = tio.ScalarImage(tensor=img)
            images.update({str(i): img2})
        img_set = tio.Subject(images)
        res = compose_transform.apply_transform(img_set)

        res_numpy = [res[k].data.numpy().copy() for k in images.keys()]
        return res_numpy
