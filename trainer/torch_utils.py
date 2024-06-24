import importlib
import os
import shutil
import numpy as np
import torch
from torch import optim
from collections import OrderedDict
from typing import Dict
from torch import nn
from .utils import get_logger

get_runtime_logger = get_logger

logger = get_logger(__name__)

def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict', strict=True):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key], strict=strict)

    if optimizer is not None and optimizer_key in state:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    import h5py
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_number_of_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class IntermediateLayerGetterDeepSearch(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    references: torchvision.models._utils.py - IntermediateLayerGetter

    """
    _version = 2
    __annotations__ ={
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers, parent='', sep='_'):
        return_layers = {sep.join([parent, str(k)]):str(v) for k, v in return_layers.items()}
        orig_return_layers = dict(return_layers)
        layers = OrderedDict()
        collect_some_layers(model, return_layers, layers, parent=parent, sep=sep)

        super(IntermediateLayerGetterDeepSearch, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x

        return out



def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        # else:
        elif batch.ndim > 1:
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))
        else:
            pass

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances

    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind:ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind:ind + 1, ...])

    return np.stack(result, axis=0)


def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return DefaultTensorboardFormatter()

    class_name = formatter_config['name']
    m = importlib.import_module('teethnet.models.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**formatter_config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    # assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    optim_name = optimizer_config.get('name', 'adam')
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)
    momentum = optimizer_config.get('momentum', 0.1)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              # nesterov=config.TRAIN.NESTEROV,
                              )
    else:
        raise NotImplementedError(optim_name)

    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    lr_config = {**lr_config}
    class_name = lr_config.get('name', 'ReduceLROnPlateau')
    # m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = get_class(class_name,
              [
                  'torch.optim.lr_scheduler'
              ])
    # clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def get_shape(some_val):
    if issubclass(type(some_val), (torch.Tensor, np.ndarray)):
        return some_val.shape
    elif issubclass(type(some_val), dict):
        return {k: get_shape(v) for k, v in some_val.items()}
    elif issubclass(type(some_val), (list, tuple)):
        return [get_shape(v) for v in some_val]
    else:
        return 'unknow type:{}'.format(type(some_val))


def get_contiguous(some_val):
    if issubclass(type(some_val), (torch.Tensor, np.ndarray)):
        if isinstance(some_val, torch.Tensor):
            return some_val.is_contiguous()
        elif isinstance(some_val, np.ndarray):
            # C-contiguous or Fortran-contiguous
            return some_val.flags['C_CONTIGUOUS'] or some_val.flags['F_CONTIGUOUS']
        else:
            raise ValueError('invalid type')
    elif issubclass(type(some_val), dict):
        return {k: get_contiguous(v) for k, v in some_val.items()}
    elif issubclass(type(some_val), (list, tuple)):
        return [get_contiguous(v) for v in some_val]
    else:
        return 'unknow type:{}'.format(type(some_val))


def to_torch_tensor(x):
    if issubclass(type(x), (tuple, list)):
        return [to_torch_tensor(v) for v in x]
    elif type(x) == dict:
        return {k: to_torch_tensor(v) for k, v in x.items()}
    elif type(x) == np.ndarray:
        return torch.from_numpy(x)
    elif type(x) == torch.Tensor:
        return x
    elif isinstance(x, (str,)):
        return x
    elif np.isscalar(x):
        return torch.Tensor([float(x)])
    else:
        raise NotImplementedError('not implemented for conversion as tensor:{}'.format(type(x)))

def squeeze(tensor_or_ndarray):
    if isinstance(tensor_or_ndarray, np.ndarray):
        return np.squeeze(tensor_or_ndarray)
    elif issubclass(type(tensor_or_ndarray), (tuple, list)):
        return [squeeze(v) for v in tensor_or_ndarray]
    elif issubclass(type(tensor_or_ndarray), torch.Tensor):
        return torch.squeeze(tensor_or_ndarray)
    elif type(tensor_or_ndarray) == dict:
        return {k: to_numpy(v, squeeze) for k, v in tensor_or_ndarray.items()}
    else:
        raise NotImplemented('what valud:{}'.format(type(tensor_or_ndarray)))


def to_numpy(tensor_or_ndarray, squeeze=True):
    if isinstance(tensor_or_ndarray, np.ndarray):
        return tensor_or_ndarray
    elif issubclass(type(tensor_or_ndarray), (tuple, list)):
        return [to_numpy(v, squeeze) for v in tensor_or_ndarray]
    elif issubclass(type(tensor_or_ndarray), torch.Tensor):
        if squeeze:
            return np.squeeze(tensor_or_ndarray.detach().cpu().numpy())
        else:
            return tensor_or_ndarray.detach().cpu().numpy()
    elif type(tensor_or_ndarray) == dict:
        return {k: to_numpy(v, squeeze) for k, v in tensor_or_ndarray.items()}
    else:
        logger.error('what valud:{}'.format(type(tensor_or_ndarray)))
        # raise NotImplemented()
        return tensor_or_ndarray


def to_device(x, name):
    if type(x) == list:
        return [to_device(v, name) for v in x]
    elif type(x) == torch.Tensor:
        if x.device.type == name.type:
            return x
        else:
            return x.to(name)
    elif type(x) == dict:
        return {k: to_device(v, name) for k, v in x.items()}
    elif type(x) == np.ndarray:
        # logger = get_runtime_logger()
        # logger.warn('to try numpy-array-device-set')
        return x
    elif isinstance(x, (str,)):
        return x
    else:
        raise NotImplementedError('not implemented for conversion as tensor')


def to_batch(x):
    if issubclass(type(x), (tuple, list)):
        return [to_batch(v) for v in x]
    elif type(x) == dict:
        return {k: to_batch(v) for k, v in x.items()}
    elif type(x) == np.ndarray:
        return x[np.newaxis]
    elif type(x) == torch.Tensor:

        return x[None]
    elif isinstance(x, (str,)):
        return x
    else:
        raise NotImplementedError('not implemented for conversion as tensor:{}'.format(type(x)))


def to_torch_type(type_name):
    table = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,

    }
    return table[type_name]


def to_type(x, dtype):
    """
    if dtype float 32
        float32 or float 64 ------> float32
        int32, int64 ---> not changed
    if dtype int 32
        float32 or float 64 ------>  not changed
        int32, int64 -----> int32

    Args:
        x ():
        dtype (str): 'int32' 'int64' 'float32' 'float64'

    Returns:

    """
    if issubclass(type(x), (tuple, list)):
        return [to_type(v, dtype) for v in x]
    elif type(x) == dict:
        return {k: to_type(v, dtype) for k, v in x.items()}
    elif type(x) == np.ndarray:
        # 정수형일고 일치할때만
        if np.issubdtype(x.dtype, np.integer) and np.issubdtype(dtype, np.integer):
            return x.astype(dtype)
        # 실수형일때
        elif np.issubdtype(x.dtype, np.floating) and np.issubdtype(dtype, np.floating):
            return x.astype(dtype)
        else:
            return x
        # return x.astype(dtype)
    elif isinstance(x, torch.Tensor):

        if x.dtype.is_floating_point:
            if dtype in ['float32', 'float64']:
                torch_type = to_torch_type(dtype)
            elif dtype in [torch.float16, torch.float32, torch.float64]:
                torch_type = dtype
            else:
                return x

            return x.to(torch_type)
        elif not x.dtype.is_floating_point:
            if dtype in ['int32', 'int64']:
                torch_type = to_torch_type(dtype)
            elif dtype in [torch.int16, torch.int32, torch.int64]:
                torch_type = dtype
            else:
                return x
            return x.to(torch_type)
        else:
            return x
        # else:
        #     raise NotImplementedError('not implemented:{}/{}'.format(type(x), x.dtype))
    elif isinstance(x, (str,)):
        return x
    else:
        raise NotImplementedError('not implemented for conversion as tensor:{}'.format(type(x)))


def data_convert(datas, **kwargs):
    """
    kwargs:
        device: torch.device
        dtype: str, data type
        batch: bool, if expansion batch
    1. to_toerch_tensor
    2. device convert
    3. make sure main dtype(float32 or float64)
    4. batch


    Returns
    -------

    """
    dtype = kwargs.get('dtype', 'float32')
    device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    batch_expansion = kwargs.get('batch', True)
    # funcs = [
    #     to_torch_tensor,
    # ]
    res = datas
    res = to_torch_tensor(res)
    res = to_device(res, device)
    res = to_type(res, dtype)
    if batch_expansion:
        res = to_batch(res)
    if isinstance(res, torch.Tensor):
        return res
    else:
        return tuple(res)

# def collect_some_layers(model: torch.nn.Module, some_layers, container, parent):
#     for name, child in model.named_children():
#         if isinstance(child, (torch.nn.Sequential, torch.nn.ModuleDict, torch.nn.ModuleList)):
#             # name_cat = parent + '.' + name
#             splits_names = [ln.split('.') for ln in some_layers.keys()]
#
#             if [(not name in splits) or (splits.index(name) == len(splits)-1) for splits in splits_names]:
#                 print(name)
#             else:
#                 pass
#
#             # for name in splits_names:
#             #     name
#             # splits_names
#
#             # container[name] = child
#
#             # if name in some_layers:
#             #     del some_layers[name]
#         else:
#             pass
#             # if not return_layers:
#             #     break
#         # if name in some_layers:
#         #
#         #     if name in some_layers:
#         #         del some_layers[name]
#         #     if not some_layers:
#         #         break
#             # container.append(child)
#         collect_some_layers(child, some_layers, container, parent + '.' + name)

def collect_some_module(model: torch.nn.Module, some_module, continaer):
    for name, child in model.named_children():
        if isinstance(child, some_module):
            continaer.append(child)
        collect_some_module(child, some_module, continaer)



def print_some_module(model: torch.nn.Module, parent):
    for name, child in model.named_children():
        if isinstance(child, (nn.Sequential, nn.ModuleDict, nn.ModuleList, torch.nn.Module)):
            print(parent + '.' + name, ':', type(child))
        # else:
        #     print(name, type(child))
            pass

        print_some_module(child, parent + '.' + name)

def get_last_module(model, parent='', sep='.', seq=[]):
    for name, child in model.named_children():
        concat_name = parent + sep + name
        seq.append(concat_name)
        get_last_module(child, concat_name, sep, seq)

    return seq
    # return concat_name


# def print_some_module(model: torch.nn.Module, parent):
#     for name, child in model.named_children():
#         if isinstance(child, (nn.Sequential, nn.ModuleDict, nn.ModuleList)):
#             print(parent + '.' + name, ':', type(child))
#         else:
#             # print(name, type(child))
#             pass
#
#         print_some_module(child, parent + '.' + name)


def collect_some_layers(model: torch.nn.Module, some_layers, container, parent, sep='_'):
    """
    재귀함수로 구현
    module -
    Parameters
    ----------
    model :
    some_layers :
    container :
    parent :
    sep :

    Returns
    -------

    """
    if not some_layers:
        # 재귀함수로 동작하기 때문에, 제일 앞에서 체크
        return
    for name, child in model.named_children():
        cat_name = sep.join([parent, name])
        # len([c for c in child.children()])
        has_child = len([c for c in child.children()]) > 0
        if isinstance(child, (torch.nn.Sequential, torch.nn.ModuleDict, torch.nn.ModuleList)) and \
                (isinstance(child, torch.nn.Module) and has_child):
            container[cat_name] = child
            if cat_name in some_layers:
                # container[cat_name] = child
                some_layers.pop(cat_name)
            else:
                pass
        elif isinstance(child, torch.nn.Module):
            container[cat_name] = child
            if cat_name in some_layers:
                some_layers.pop(cat_name)
        else:
            pass
                # container
            # torch.nn.module
            # container[cat_name] = child
            pass
        collect_some_layers(child, some_layers, container, cat_name)

def norm_boxes(boxes, shape):
    scale = torch.asarray(shape)
    scale = scale.repeat(1, 2)
    return (boxes / scale).to(torch.float32)
    # return np.divide(boxes, scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    scale = torch.asarray(shape)
    scale = scale.repeat(1, 2)
    return boxes * scale
    # return np.multiply(boxes, scale)


def set_dropout(model:torch.nn.Module, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)




def load_weights(model: torch.nn.Module, param_filename, weight_key='model_state_dict', perform_keys = ['dsc', 'sen', 'ppv']):
    """

    Args:
        model (): 읽을 모델
        param_filename (str):  checkpiont file
        perform_key (str): 비교할 performance key

    Returns:
        checkpoint['valid']

    """
    logger = get_runtime_logger()
    if os.path.exists(param_filename):
        checkpoint = torch.load(param_filename, map_location=torch.device('cuda'))

        for key in perform_keys:
            performs = checkpoint.get('valid', {}).get(key, [])
            if len(performs) > 0:
                logger.info(f'{key}, {np.mean(performs)}')

        try:
            model.load_state_dict(checkpoint[weight_key], strict=False)
        except Exception as e:
            logger.error(e)
        logger.info('loading complete:{}'.format(param_filename))
        return checkpoint
    else:
        logger.error('cannot load weights file:{}'.format(param_filename))
        return dict()

