import torch
from torch.nn import Module
import importlib
import os.path
import pickle
import numpy as np
from skimage import measure
from skimage.metrics import adapted_rand_error, peak_signal_noise_ratio, mean_squared_error

from .losses import compute_per_channel_dice
from . import torch_utils
from .torch_utils import expand_as_one_hot, convert_to_numpy
from .utils import get_logger, create_class, get_class
from .torch_utils import expand_as_one_hot


logger = get_logger(__name__)


def _create_metric(name, param:dict):
    model_class = get_class(name, modules=[
        'trainer.test.testmodel',
        'interfaces.pidnetmodel',
        'trainer.metrics',
    ])
    return create_class(model_class, param)


def get_evaluation_metric(config):
    metrics_config = config.get('metrics', {}) or config.get('eval_metric', {})
    names = metrics_config['names']

    res = list()
    for name in names:
        res.append(
            _create_metric(name, metrics_config.get(name, {}))
        )
    return res

#
# class Accuracy:
#     """
#     Computes accuracy between ground truth and predicted segmentation a a given threshold value.
#     Defined as: AC = TP / (TP + FP + FN).
#     Kaggle DSB2018 calls it Precision, see:
#     https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric.
#     """
#
#     def __init__(self, iou_threshold):
#         self.iou_threshold = iou_threshold
#
#     def __call__(self, input_seg, gt_seg):
#         metrics = SegmentationMetrics(gt_seg, input_seg).metrics(self.iou_threshold)
#         return metrics['accuracy']
#
#
# class AveragePrecision:
#     """
#     Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
#     https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
#     """
#
#     def __init__(self):
#         self.iou_range = np.linspace(0.50, 0.95, 10)
#
#     def __call__(self, input_seg, gt_seg):
#         # compute contingency_table
#         sm = SegmentationMetrics(gt_seg, input_seg)
#         # compute accuracy for each threshold
#         acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
#         # return the average
#         return np.mean(acc)


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        # assert input.dim() == 5
        assert input.dim() <= target.dim() + 1, 'input greater than 1 or same'

        n_classes = input.size()[1]

        if input.dim() > target.dim():
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)



class DSC:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target, dim=1):
        '''
        inputs:
            y_pred [n_classes, x, y, z] probability
            y_true [n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''


        n_classes = input.size()[1]

        # if input.dim() > target.dim():
        if input.dim() > target.dim():
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        y_true = target
        y_pred = input

        smooth = 1e-5
        mdsc = torch.tensor(0.0).to(y_pred.device)
        # n_classes = y_pred.shape[-1]

        # convert probability to one-hot code
        max_idx = torch.argmax(y_pred, dim=dim, keepdim=True)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(dim, max_idx, 1)
        w = 1./n_classes
        counts = 0
        for c in range(1, n_classes):
            pred_flat = one_hot[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            # w = class_weights[c] / class_weights.sum()
            if true_flat.sum() > 0:
                counts += 1
                mdsc += ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))

        if counts > 0:
            mdsc /= counts

        return mdsc


class SEN:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target, dim=1):

        '''
        inputs:
            y_pred [n_classes, x, y, z] probability
            y_true [n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''

        n_classes = input.size()[1]

        if input.dim() > target.dim():
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        y_true = target
        y_pred = input

        smooth = 1.
        msen = torch.tensor(0.0).to(y_pred.device)

        # convert probability to one-hot code
        max_idx = torch.argmax(y_pred, dim=dim, keepdim=True)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(dim, max_idx, 1)
        w = 1./n_classes
        counts = 0

        for c in range(1, n_classes):
            pred_flat = one_hot[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            # w = class_weights[c] / class_weights.sum()

            if true_flat.sum() > 0:
                counts += 1
                msen += ((intersection + smooth) / (true_flat.sum() + smooth))

        if counts > 0:
            msen /= counts

        return msen



class PPV:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target, dim=1):
        '''
        inputs:
            y_pred [n_classes, x, y, z] probability
            y_true [n_classes, x, y, z] one-hot code
            dim : int for chanael dimension
            class_weights
            smooth = 1.0
        '''

        n_classes = input.size()[1]


        if input.dim() > target.dim():
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        y_true = target
        y_pred = input

        smooth = 1.
        mppv = torch.tensor(0.0).to(y_pred.device)
        # n_classes = y_pred.shape[-1]

        # convert probability to one-hot code
        max_idx = torch.argmax(y_pred, dim=dim, keepdim=True)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(dim, max_idx, 1)
        w = 1./n_classes
        counts = 0

        for c in range(1, n_classes):
            pred_flat = one_hot[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            # w = class_weights[c] / class_weights.sum()
            # mppv += w * ((intersection + smooth) / (pred_flat.sum() + smooth))

            if pred_flat.sum() > 0:
                counts += 1
                mppv += ((intersection + smooth) / (pred_flat.sum() + smooth))

                # mppv += ((intersection + smooth) / (true_flat.sum() + smooth))

        if counts > 0:
            mppv /= counts

        return mppv



class Precision(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Precision, self).__init__()
        self.dim = kwargs.get('dim', -1)

    def forward(self, pred, target):
        pred_label = torch.argmax(pred, dim=self.dim)
        target_label = target
        # target_label = torch.argmax(target, dim=self.dim)
        equal = pred_label == target_label
        prec = equal.sum() / equal.numel()
        return prec





class Precision2(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Precision2, self).__init__()
        self.dim = kwargs.get('dim', -1)

    def forward(self, pred, target):
        pred_label = torch.argmax(pred, dim=self.dim)
        target_label = target[0]
        # target_label = torch.argmax(target, dim=self.dim)
        equal = pred_label == target_label
        prec = equal.sum() / equal.numel()
        return prec




class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class AdaptedRandError:
    """
    A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
    (http://brainiac2.mit.edu/SNEMI3D/evaluation).

    This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_segm()`)
    and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
    it's enough to extend this class and implement the `input_to_segm` method.

    Args:
        use_last_target (bool): if true, use the last channel from the target to compute the ARand, otherwise the first.
    """

    def __init__(self, use_last_target=False, ignore_index=None, **kwargs):
        self.use_last_target = use_last_target
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        Compute ARand Error for each input, target pair in the batch and return the mean value.

        Args:
            input (torch.tensor):  5D (NCDHW) output from the network
            target (torch.tensor): 5D (NCDHW) ground truth segmentation

        Returns:
            average ARand Error across the batch
        """

        # converts input and target to numpy arrays
        input, target = convert_to_numpy(input, target)
        if self.use_last_target:
            target = target[:, -1, ...]  # 4D
        else:
            # use 1st target channel
            target = target[:, 0, ...]  # 4D

        # ensure target is of integer type
        target = target.astype(np.int32)

        if self.ignore_index is not None:
            target[target == self.ignore_index] = 0

        per_batch_arand = []
        for _input, _target in zip(input, target):
            if np.all(_target == _target.flat[0]):  # skip ARand eval if there is only one label in the patch due to zero-division
                logger.info('Skipping ARandError computation: only 1 label present in the ground truth')
                per_batch_arand.append(0.)
                continue

            # convert _input to segmentation CDHW
            segm = self.input_to_segm(_input)
            assert segm.ndim == 4

            # compute per channel arand and return the minimum value
            per_channel_arand = [adapted_rand_error(_target, channel_segm)[0] for channel_segm in segm]
            per_batch_arand.append(np.min(per_channel_arand))

        # return mean arand error
        mean_arand = torch.mean(torch.tensor(per_batch_arand))
        logger.info(f'ARand: {mean_arand.item()}')
        return mean_arand

    def input_to_segm(self, input):
        """
        Converts input tensor (output from the network) to the segmentation image. E.g. if the input is the boundary
        pmaps then one option would be to threshold it and run connected components in order to return the segmentation.

        :param input: 4D tensor (CDHW)
        :return: segmentation volume either 4D (segmentation per channel)
        """
        # by deafult assume that input is a segmentation volume itself
        return input


class BoundaryAdaptedRandError(AdaptedRandError):
    """
    Compute ARand between the input boundary map and target segmentation.
    Boundary map is thresholded, and connected components is run to get the predicted segmentation
    """

    def __init__(self, thresholds=None, use_last_target=True, ignore_index=None, input_channel=None, invert_pmaps=True,
                 save_plots=False, plots_dir='.', **kwargs):
        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, save_plots=save_plots,
                         plots_dir=plots_dir, **kwargs)

        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel
        self.invert_pmaps = invert_pmaps

    def input_to_segm(self, input):
        if self.input_channel is not None:
            input = np.expand_dims(input[self.input_channel], axis=0)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # threshold probability maps
                predictions = predictions > th

                if self.invert_pmaps:
                    # for connected component analysis we need to treat boundary signal as background
                    # assign 0-label to boundary mask
                    predictions = np.logical_not(predictions)

                predictions = predictions.astype(np.uint8)
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label(predictions, background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAdaptedRandError(AdaptedRandError):
    def __init__(self, input_channels, thresholds=None, use_last_target=True, ignore_index=None, invert_channels=None,
                 **kwargs):

        super().__init__(use_last_target=use_last_target, ignore_index=ignore_index, **kwargs)
        assert isinstance(input_channels, list) or isinstance(input_channels, tuple)
        self.input_channels = input_channels
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        if invert_channels is None:
            invert_channels = []
        self.invert_channels = invert_channels

    def input_to_segm(self, input):
        # pick only the channels specified in the input_channels
        results = []
        for i in self.input_channels:
            c = input[i]
            # invert channel if necessary
            if i in self.invert_channels:
                c = 1 - c
            results.append(c)

        input = np.stack(results)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label((predictions > th).astype(np.uint8), background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


class GenericAveragePrecision:
    def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target
        assert metric in ['ap', 'acc']
        if metric == 'ap':
            # use AveragePrecision
            self.metric = AveragePrecision()
        else:
            # use Accuracy at 0.5 IoU
            self.metric = Accuracy(iou_threshold=0.5)

    def __call__(self, input, target):
        if target.dim() == 5:
            if self.use_last_target:
                target = target[:, -1, ...]  # 4D
            else:
                # use 1st target channel
                target = target[:, 0, ...]  # 4D

        input1 = input2 = input
        multi_head = isinstance(input, tuple)
        if multi_head:
            input1, input2 = input

        input1, input2, target = convert_to_numpy(input1, input2, target)

        batch_aps = []
        i_batch = 0
        # iterate over the batch
        for inp1, inp2, tar in zip(input1, input2, target):
            if multi_head:
                inp = (inp1, inp2)
            else:
                inp = inp1

            segs = self.input_to_seg(inp, tar)  # expects 4D
            assert segs.ndim == 4
            # convert target to seg
            tar = self.target_to_seg(tar)

            # filter small instances if necessary
            tar = self._filter_instances(tar)

            # compute average precision per channel
            segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

            logger.info(f'Batch: {i_batch}. Max Average Precision for channel: {np.argmax(segs_aps)}')
            # save max AP
            batch_aps.append(np.max(segs_aps))
            i_batch += 1

        return torch.tensor(batch_aps).mean()

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 0-index
        :param input: input instance segmentation
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    input[input == label] = 0
        return input

    def input_to_seg(self, input, target=None):
        raise NotImplementedError

    def target_to_seg(self, target):
        return target


class BlobsAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BlobsBoundaryAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction, boundary prediction and ground truth instance segmentation.
    Segmentation mask is computed as (P_mask - P_boundary) > th followed by a connected component
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds

    def input_to_seg(self, input, target=None):
        # input = P_mask - P_boundary
        input = input[0] - input[1]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class BoundaryAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given boundary prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True)
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            seg = measure.label(np.logical_not(input > th).astype(np.uint8), background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class PSNR:
    """
    Computes Peak Signal to Noise Ratio. Use e.g. as an eval metric for denoising task
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return peak_signal_noise_ratio(target, input)


class MSE:
    """
    Computes MSE between input and target
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return mean_squared_error(input, target)



def _create_metric_class(name, eval_config):
    model_class = get_class(name, modules=[
        'trainer.test.testmodel',
        'trainer.losses',
        'interfaces.pidnetmodel',
    ])
    return create_class(model_class, eval_config)

#
# def get_evaluation_metric(config):
#     """
#     Returns the evaluation metric function based on provided configuration
#     :param config: (dict) a top level configuration object containing the 'eval_metric' key
#     :return: an instance of the evaluation metric
#     """
#
#     # def _metric_class(class_name):
#     #     m = importlib.import_module('teethnet.models.unet3d.metrics')
#     #     clazz = getattr(m, class_name)
#     #     return clazz
#
#     assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
#     metric_config = config['eval_metric']
#     # multiple_metric = metric_config.get('multiple', True)
#
#     metric_class = _create_metric_class(metric_config, metric_config)
#     return metric_class(**metric_config)


def get_evaluation_multiple_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    # def _metric_class(class_name, names):
    #     m = importlib.import_module('teethnet.models.unet3d.metrics')
    #     clazz = getattr(m, class_name)
    #     return clazz

    def get_class(class_name, modules):
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz
        raise RuntimeError(f'Unsupported metric class: {class_name}')

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    multiple_metric = metric_config.get('multiple', True)
    # metric_class = _metric_class(metric_config['names'])
    if multiple_metric:
        classez = []
        # for nmae
        for name in metric_config['names']:
            # cls = _metric_class(name)
            cls = get_class(name, [
                'teethnet.models.unet3d.metrics',
                'teethnet.models.pytorchyolo3d.utils.metric',
            ])
            classez.append(cls(**metric_config.get(name, {})))
        return classez
    else:
        cls = get_evaluation_metric(config)
        return [cls]

    # metric_class = _metric_class(metric_config['name'])
    # return metric_class(**metric_config)




class DSC:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target, dim=1):
        '''
        inputs:
            y_pred [n_classes, x, y, z] probability
            y_true [n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''

        # assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        y_true = target
        y_pred = input

        smooth = 1e-5
        mdsc = torch.tensor(0.0).to(y_pred.device)
        # n_classes = y_pred.shape[-1]

        # convert probability to one-hot code
        max_idx = torch.argmax(y_pred, dim=dim, keepdim=True)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(dim, max_idx, 1)
        w = 1./n_classes
        counts = 0
        for c in range(1, n_classes):
            pred_flat = one_hot[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            # w = class_weights[c] / class_weights.sum()
            if true_flat.sum() > 0:
                counts += 1
                mdsc += ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))

        if counts > 0:
            mdsc /= counts

        return mdsc

class SEN:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target, dim=1):

        '''
        inputs:
            y_pred [n_classes, x, y, z] probability
            y_true [n_classes, x, y, z] one-hot code
            class_weights
            smooth = 1.0
        '''
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        y_true = target
        y_pred = input

        smooth = 1.
        msen = torch.tensor(0.0).to(y_pred.device)

        # convert probability to one-hot code
        max_idx = torch.argmax(y_pred, dim=dim, keepdim=True)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(dim, max_idx, 1)
        w = 1./n_classes
        counts = 0

        for c in range(1, n_classes):
            pred_flat = one_hot[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            # w = class_weights[c] / class_weights.sum()

            if true_flat.sum() > 0:
                counts += 1
                msen += ((intersection + smooth) / (true_flat.sum() + smooth))

        if counts > 0:
            msen /= counts

        return msen



class PPV:
    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target, dim=1):
        '''
        inputs:
            y_pred [n_classes, x, y, z] probability
            y_true [n_classes, x, y, z] one-hot code
            dim : int for chanael dimension
            class_weights
            smooth = 1.0
        '''
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        y_true = target
        y_pred = input

        smooth = 1.
        mppv = torch.tensor(0.0).to(y_pred.device)
        # n_classes = y_pred.shape[-1]

        # convert probability to one-hot code
        max_idx = torch.argmax(y_pred, dim=dim, keepdim=True)
        one_hot = torch.zeros_like(y_pred)
        one_hot.scatter_(dim, max_idx, 1)
        w = 1./n_classes
        counts = 0

        for c in range(1, n_classes):
            pred_flat = one_hot[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            # w = class_weights[c] / class_weights.sum()
            # mppv += w * ((intersection + smooth) / (pred_flat.sum() + smooth))

            if pred_flat.sum() > 0:
                counts += 1
                mppv += ((intersection + smooth) / (pred_flat.sum() + smooth))

                # mppv += ((intersection + smooth) / (true_flat.sum() + smooth))

        if counts > 0:
            mppv /= counts

        return mppv


class SegmentOffsetMetric(Module):
    def __init__(self, **kwargs):
        super(SegmentOffsetMetric, self).__init__()
        # mesh-voxelizer 같이 sparse한 볼륨에 대해 metric 예외처리할지 옵션
        self.mask_segment_applied = kwargs.get('mask_segment_applied', False)
        self.dice = DSC()
        self.seg_threshold = .5
        self.offset_threshold = [1, 2, 3]

    def forward(self, pred, target):
        pred_seg, pred_offset = pred if len(pred) == 2 else pred[0]
        target_seg, t_offset = target
        t_seg, t_mask = target_seg.split(1, dim=1)

        t_seg_squeeze = t_seg.squeeze(1)
        n_classes = pred_seg.size(1)

        if self.mask_segment_applied:
            pred_seg = torch.where(t_mask > 0, pred_seg, torch.zeros_like(pred_seg))

        t_seg_onehot = torch.nn.functional.one_hot(t_seg_squeeze, n_classes).permute(
            [0, t_seg_squeeze.ndim, *tuple(range(1, t_seg_squeeze.ndim))]).contiguous().to(pred_seg.dtype)

        # offset = (seg > self.seg_threshold ).to(offset.dtype) * offset
        seg_dice = self.dice(pred_seg, t_seg_onehot)

        # (
        ijk = torch.meshgrid(*[torch.arange(i, device=t_offset.device) for i in t_offset.shape[2:]], indexing='ij')
        # and batch expansion
        ijk = torch.stack(ijk, dim=0)[None]

        coords_scale = torch.tensor(pred_seg.shape[2:], device=pred_seg.device)
        coords_scale = coords_scale.reshape([-1, *[1 for _ in range(3)]])

        pred_center = ijk + pred_offset * coords_scale
        gt_center = ijk + t_offset * coords_scale
        dists = torch.linalg.norm(pred_center - gt_center, dim=1, keepdim=True)
        # t_seg_prob = t_seg[:, 1:]

        t_seg_dists = dists[t_seg == 1]
        pred_seg_dists = dists[torch.argmax(pred_seg, dim=1, keepdim=True) == 1]
        accs = {}
        for thres in self.offset_threshold:
            # tgt_mask_acc = (t_seg_dists < thres).sum() / t_seg_dists.numel()
            pred_mask_acc = (pred_seg_dists < thres).sum() / pred_seg_dists.numel()
            # accs.update({f'taget_offset_acc_{thres}': tgt_mask_acc})
            accs.update({f'pred_offset_acc_{thres}': pred_mask_acc})

        if False:
            # from tools import torch_utils
            # import time
            p0, q0 = pred_center[..., t_seg[0, 0] > 0], (pred_offset * coords_scale)[..., t_seg[0, 0] > 0]
            p1, q1 = torch_utils.to_numpy([p0, q0])
            savename = 'd:/temp/pred_center'
            os.makedirs(savename, exist_ok=True)
            with open(os.path.join(savename, f'{time.strftime("%Y%m%d%H%M%S")}.pkl'), 'wb') as f:
                pickle.dump({
                    'center': p1,
                    'offset': q1
                }, f)

        return {
            'seg_dice': seg_dice,
            **accs
        }

        # from tools import vtk_utils, torch_utils
        # x = torch_utils.to_numpy(gt_center.permute([2, 3, 4, 0, 1]))
        # y = torch_utils.to_numpy(t_seg > 0)
        # xy = x[y]
        # vtk_utils.show_actors([xy])

class SegmentOffsetMetric2(Module):
    def __init__(self, **kwargs):
        super(SegmentOffsetMetric2, self).__init__()
        self.seg_offset_metric = SegmentOffsetMetric(**kwargs)

    def forward(self, preds, targets):

        losses = {}
        for i, (y_pred, y_target) in enumerate(zip(preds, targets)):
            prefix = f'{i:02d}_'
            loss = self.seg_offset_metric(y_pred, y_target)
            losses.update({
                prefix + k: v for k, v in loss.items()
            })
        return losses



class Precision(Module):
    def __init__(self, **kwargs):
        super(Precision, self).__init__()
        self.dim = kwargs.get('dim', 1)

    def forward(self, pred, target):
        pred_label = torch.argmax(pred, dim=self.dim)
        target_label = torch.argmax(target, dim=self.dim)
        equal = pred_label == target_label
        prec = equal.sum() / equal.numel()
        return prec


class SegmentClassificationMetric(Module):
    def __init__(self, **kwargs):
        super(SegmentClassificationMetric, self).__init__()

    def forward(self, preds, targets):

        preds, targets = torch_utils.squeeze((preds, targets))
        pred_mask, pred_classes = preds
        target_mask, target_class = targets

        metrics = {
            'dsc': DSC(),
            'ppv': PPV(),
            'sen': SEN(),
        }
        seg_metrics = {name: eval(pred_mask, target_mask) for name, eval in metrics.items()}


        pred_label = torch.argmax(pred_classes, dim=1)
        size = min(pred_label.size(0), pred_classes.size(0))
        # target_label = torch.argmax(target, dim=self.dim)
        equal = pred_label[:size] == target_class[:size]

        num_actual = pred_mask.shape[0]
        num_stat = max(pred_label.shape[0] - num_actual, 1)
        # prec = equal.sum() / equal.numel()
        pred_actual = equal[:num_actual].sum() / num_actual
        pred_stat = equal[num_actual:].sum() / num_stat
        return {
            **seg_metrics,
            'roi_class_precision': pred_actual,
            'stat_class_precision': pred_stat,

        }
