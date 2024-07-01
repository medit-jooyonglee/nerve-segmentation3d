import torch
import collections
from torch.utils.data import DataLoader, Dataset
from trainer.utils import get_logger, get_class
from typing import Callable
logger = get_logger(__name__)


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def initialize_detection_model(self, some_model):
        # 학습 데이터 배치를 위한 필요한 detection model 초기화
        pass

    def detection(self, inputs, post_func:Callable[[any], any]=None):
        pass
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)

    def change_mode(self):
        pass

    def train(self, true_or_false):
        pass


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def _loader_classes(name):
    return get_class(name, [
        # 'trainer.test.testmodel',
        # 'interfaces.pidnetmodel',
        # 'patchset',
        # 'dataset.loader',
        'dataset.nervedataset',
        'dataset.miccaitoothfairy',
        # 'vit_pytorch.vit',
    ])


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'valid': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('name', None) or loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    assert all([('file_paths' in loaders_config[name]) for name in ['train', 'valid']]), f'not defined "file_paths", {loaders_config}'
    train_file_paths = loaders_config['train']['file_paths']
    valid_file_paths = loaders_config['valid']['file_paths']

    train_datasets = dataset_class(**{**loaders_config, 'paths': train_file_paths, 'name': 'train'})
    valid_datasets = dataset_class(**{**loaders_config, 'paths': valid_file_paths, 'name': 'valid'})
    # val_datasets = dataset_class(loaders_config)
    assert len(train_datasets) > 0, 'emtpy train dataset'
    assert len(valid_datasets) > 0, 'emtpy valid dataset'

    num_workers = loaders_config.get('num_workers', 0)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config.get('device') == 'cpu' and False:
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')


    return {
        'train': DataLoader(train_datasets, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'valid': DataLoader(valid_datasets, batch_size=batch_size, shuffle=False, pin_memory=True,
                          num_workers=num_workers),
    }


