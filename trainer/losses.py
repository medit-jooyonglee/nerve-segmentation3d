import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from typing import Dict

import numpy as np


from .utils import get_logger, get_class, create_class
from .torch_utils import expand_as_one_hot


logger = get_logger(__name__)
logger.info('test')


class CrossEntropyLossSq(nn.Module):
    def __init__(self):
        super(CrossEntropyLossSq, self).__init__()
        self.loss = CrossEntropyLoss()

    def forward(self, input, target):
        return self.loss(input[0], target[0])


class CrossEntropyLossSq2(nn.Module):
    def __init__(self):
        super(CrossEntropyLossSq2, self).__init__()
        self.loss = CrossEntropyLoss()

    def forward(self, input, target):
        return self.loss(input[0], target[0][0])




class BalanceMeanDiceLoss(nn.Module):
    def __init__(self):
        super(BalanceMeanDiceLoss, self).__init__()
        self.posit_negat_ratio = 3

    def forward(self, input, target, smooth=1.0):
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            positive = target

            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()
        # BCEDiceLoss
        #
        # loss = 0.
        # for c in range(0, n_classes):
        #     pred_flat = input[:, c].reshape(-1)
        #     true_flat = target[:, c].reshape(-1)
        #     intersection = (pred_flat * true_flat).sum()
        #
        #     # with weight
        #     # w = class_weights[c] / class_weights.sum()
        #
        #     flat_sum = pred_flat.sum() + true_flat.sum() + smooth
        #
        #     loss += (1 - ((2. * intersection + smooth) / flat_sum)) if flat_sum != 0 else 0.0
        # return loss


class MeanDiceLoss(nn.Module):
    """
    Compute average dice loss
    """

    def __init__(self, **kwargs):
        super(MeanDiceLoss, self).__init__()
        pass

    def forward(self, input, target, smooth=1.0):

        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes)
            # target = one_hot(target, n_classes)
        assert input.size() == target.size()

        loss = 0.
        for c in range(0, n_classes):
            pred_flat = input[:, c].reshape(-1)
            true_flat = target[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()

            # with weight
            # w = class_weights[c] / class_weights.sum()

            flat_sum = pred_flat.sum() + true_flat.sum() + smooth

            loss += (1 - ((2. * intersection + smooth) / flat_sum)) if flat_sum != 0 else 0.0
        return loss


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    try:
        input = flatten(input)
        target = flatten(target)
    except:
        pass
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


# TODO: It doesn't work now
# Use BCEDiceLoss
class BCEDiceLossBalanced(nn.Module):
    def __init__(self, **kwargs):
        super(BCEDiceLossBalanced, self).__init__()
        alpha, beta = kwargs.get('alpha', .5), kwargs.get('beta', .5)
        self.alpha = alpha
        # self.bce = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = DiceLoss()
        self.negative = 5.

    def forward(self, input, target, weight=None):
        # import vtk_utils
        # vtk_utils.show_actors([ *vtk_utils.auto_refinement_mask(target[0].cpu().numpy())])
        def get_balanced_mask(target, min_of_samples=25):
            import numpy as np

            num_of_samples = np.inf
            count_table = []
            if isinstance(target, torch.Tensor):
                mask = target[0].cpu().numpy().copy()
            else:
                mask = target.copy()

            for i in range(0, np.max(mask) + 1):
                count = len(np.where(mask == i)[0])
                count_table.append(count)
                if count > min_of_samples:
                    num_of_samples = np.min([num_of_samples, count]).astype(np.int32)

            for i, num in enumerate(count_table):
                if num < min_of_samples or num == num_of_samples:
                    continue

                idx = np.where(mask == i)
                mask[mask == i] = 0
                w = self.negative if i == 0 else 1
                sampling_idx = np.random.choice(count_table[i], int(num_of_samples * w), False)
                mask[idx[0][sampling_idx], idx[1][sampling_idx], idx[2][sampling_idx]] = 1

            return mask, num_of_samples

        mask, num_of_samples = get_balanced_mask(target)

        import numpy as np

        if num_of_samples == np.inf:
            num_of_samples = 0

        if target.dim() != input.dim():
            n_classes = input.size()[1]
            target = expand_as_one_hot(target, C=n_classes)

        for i in range(input.shape[1]):
            target[0][i] = torch.from_numpy(target[0][i].cpu().numpy() * mask)
            input[0][i] = torch.from_numpy(input[0][i].cpu().detach().numpy() * mask)

        # target[0] = torch.from_numpy(target[0].cpu().numpy()*mask)
        sum_bce = 0
        sum_dice = 0

        for i in range(0, input.shape[1]):
            # Positive Sample Loss Calculation
            _target = target[0][i].cpu().numpy().copy()

            # if len(_target[_target>0])> 0:
            #     _input = input[0][i].cpu().detach().numpy().copy()
            #     _input = _input[np.where(_target>0)]
            #     _target = _target[np.where(_target>0)]
            #
            #     sum_bce += self.bce(torch.from_numpy(_input),torch.from_numpy(_target))
            #     sum_dice += self.dice(torch.from_numpy(_input),torch.from_numpy(_target))

            # Negative Sample Loss Calculation
            _target = target[0][i].cpu().numpy().copy()
            idx = np.where(_target == 0)

            _input = input[0][i].cpu().detach().numpy().copy()

            # print(len(idx[0]), num_of_samples)
            sampling_idx = np.arange(len(idx[0]))
            # sampling_idx = np.random.choice(len(idx[0]), num_of_samples, False)

            _input = _input[idx[0][sampling_idx], idx[1][sampling_idx], idx[2][sampling_idx]]
            _target = _target[idx[0][sampling_idx], idx[1][sampling_idx], idx[2][sampling_idx]]

            sum_bce += self.bce(torch.from_numpy(_input), torch.from_numpy(_target))
            sum_dice += self.dice(torch.from_numpy(_input), torch.from_numpy(_target))

        sum_bce /= input.shape[1] * 2
        sum_dice /= input.shape[1] * 2

        if isinstance(sum_bce, torch.Tensor):
            sum_bce.requires_grad_(True)
        if isinstance(sum_dice, torch.Tensor):
            sum_dice.requires_grad_(True)

        return self.alpha * sum_bce + self.beta * sum_dice
        # return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, **kwargs):
        super(BCEDiceLoss, self).__init__()
        alpha, beta = kwargs.get('alpha', .5), kwargs.get('beta', .5)

        self.alpha = alpha
        # self.bce = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target, mask=None):
        if target.dim() == (input.dim() - 1):
            n_classes = input.size()[1]
            if target.dim() >= 3:
                if mask is not None and mask.shape == target.shape:
                    target = expand_as_one_hot(target, C=n_classes)
                    weight = mask[:, None].float()
                    target = target * weight
                    input = input * weight
                else:
                    target = expand_as_one_hot(target, C=n_classes)
            elif input.shape[1:] == target.shape:
                # elif input.shape[1:] == target.shape:
                target = torch.nn.functional.one_hot(target, input.shape[0]).T
                target = target.to(input.dtype)
            else:
                # onehot-class first channel
                order = [i for i in range(target.ndim)]
                target = torch.nn.functional.one_hot(target, n_classes).permute(
                    [0, target.ndim, *order[1:]]).contiguous()
                target = target.to(input.dtype)

        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class BCEDiceLossMyBalance(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta, max_negat=100):
        super(BCEDiceLossMyBalance, self).__init__()
        # self.alpha = alpha
        self.bce = BCEDiceLoss(alpha, beta)
        # self.beta = beta
        # self.dice = DiceLoss()
        self.max_negat_ratio = max_negat

    def forward(self, input, target, weight=None):
        assert target.dim() == 4
        ch = input.shape[1]
        # sample_mask = target > 0
        # sample_mask = torch.logical_or(target == 3, target == 4)
        # supress some label
        focal_weight = torch.arange(ch, dtype=target.dtype, device=target.device)
        focal_weight[:3] = 0
        focal_weight[5:] = 0
        target = focal_weight[target]
        sample_mask = target > 0

        posit_mask = sample_mask
        # negat_size = torch.sum(sample_mask)
        num_posit = torch.sum(sample_mask)
        num_negat = (num_posit * np.random.uniform(10, self.max_negat_ratio)).to(torch.int64)
        negat_mask = torch.logical_not(sample_mask)
        max_negat_size = negat_mask.sum()
        num_negat = torch.minimum(max_negat_size, num_negat)
        # negat_mask
        negat_args = torch.where(negat_mask)
        posit_args = torch.where(posit_mask)
        idx = torch.randperm(sample_mask.numel() - num_posit)[:num_negat]
        sel_negat_args = [arg[idx] for arg in negat_args]
        b, i, j, k = [torch.cat([a, b]) for a, b in zip(posit_args, sel_negat_args)]
        # sample_mask[b[idx], i[idx], j[idx], k[idx]] = True
        balance_mask = torch.zeros_like(target, dtype=torch.bool)
        balance_mask[b, i, j, k] = True

        sample_target = target[balance_mask]
        # exmask = sample_mask.expand_as(input)
        order = [i for i in range(input.ndim)]
        # (N, ch) -->(ch, N)
        sample_input = input.permute([0, *order[2:], 1])[balance_mask].transpose(1, 0)
        # s = input[0, :, sample_mask[0]]
        return self.bce(sample_input[None], sample_target[None])

        # return self.bce(input, target)
        # target = expand_as_one_hot(target, C=n_classes)
        # sample_input = input[input_sample_mask]

        # target > 0

        # return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(1)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().cuda()
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        weight = torch.tensor(weight)

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# PyTorch

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = 0.8
        self.gamma = 2.0

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, inputs, targets, smooth=1):
        alpha = self.alpha
        beta = self.beta
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1, ):
        # inputs = F.sigmoid(inputs)
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class FocalBCEFocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalBCEFocalTverskyLoss, self).__init__()
        self.focal_bce = FocalLoss()
        self.focal_tversky = FocalTverskyLoss()
        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, input, target, mask=None):
        if target.dim() != input.dim():
            if target.dim() >= 3:
                n_classes = input.size()[1]
                if mask is not None and mask.shape == target.shape:
                    target = expand_as_one_hot(target, C=n_classes)
                    weight = mask[:, None].float()
                    target = target * weight
                    input = input * weight
                else:
                    target = expand_as_one_hot(target, C=n_classes)
            elif input.shape[1:] == target.shape:
                target = torch.nn.functional.one_hot(target, input.shape[0]).T
                target = target.to(input.dtype)

        return self.alpha * self.focal_bce(input, target) + self.beta * self.focal_tversky(input, target)


class FocalBCEFocalTverskyBalanceLoss(nn.Module):
    def __init__(self):
        """
        데이터 불균형으로 학습이 전혀 안되는 싱견관 같은 부분만 별도로 bce & dice 작영
        나머지는 focalbce & tversky 그대로 적용
        """
        super(FocalBCEFocalTverskyBalanceLoss, self).__init__()
        self.focal_bce = FocalLoss()
        self.focal_tversky = FocalTverskyLoss()
        self.alpha = 0.5
        self.beta = 0.5

        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, input, target, mask=None):
        if target.dim() != input.dim():
            if target.dim() >= 3:
                n_classes = input.size()[1]
                if mask is not None and mask.shape == target.shape:
                    focal_weight = torch.arange(n_classes, dtype=target.dtype, device=target.device)
                    # focal_weight[3:5] = 1
                    focal_weight[:3] = 0
                    focal_weight[5:] = 0
                    target = focal_weight[target]
                    target = expand_as_one_hot(target, C=n_classes)
                    # weight = mask[:, None].float()

                    # focal_weight = expand_as_one_hot(focal_target, C=n_classes)
                    #
                    # target = target * weight
                    # input = input * weight

                    # focal_target = target * focal_target
                    # focal_input = input * focal_weight
                else:
                    raise ValueError
                    # target = expand_as_one_hot(target, C=n_classes)
            elif input.shape[1:] == target.shape:
                raise ValueError
                # target = torch.nn.functional.one_hot(target, input.shape[0]).T
                # target = target.to(input.dtype)
        return self.focal_bce(input, target) + self.focal_tversky(input, target)
        # return self.focal_bce(input, target) + self.focal_tversky(input, target) + \
        #        self.bce(focal_input, focal_target) + self.dice(focal_input, focal_target)


class SoftCLDice(nn.Module):
    def __init__(self, iters=6, smooth=1.):
        super(SoftCLDice, self).__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, inputs, targets):
        n_classes = inputs.size()[1]
        if targets.dim() == 4:
            targets = expand_as_one_hot(targets, C=n_classes)

        skel_pred = soft_skel(inputs[:, 1], self.iters)
        skel_true = soft_skel(targets[:, 1], self.iters)
        tprec = (torch.sum(torch.multiply(skel_pred, targets[:, 1])) + self.smooth) / (
                torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, inputs[:, 1])) + self.smooth) / (
                torch.sum(skel_true) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


class SoftDiceCLDice(SoftCLDice):
    def __init__(self, iters=5, alpha=0.1, smooth=1.):
        super(SoftCLDice, self).__init__()
        self.iters = iters
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, inputs, targets):
        n_classes = inputs.size()[1]
        if targets.dim() == 4:
            targets = expand_as_one_hot(targets, C=n_classes)

        dice = soft_dice(inputs, targets)
        skel_pred = soft_skel(inputs[:, 1], self.iters)
        skel_true = soft_skel(targets[:, 1], self.iters)
        tprec = ((torch.sum(torch.multiply(skel_pred, targets[:, 1])) + self.smooth)
                 / (torch.sum(skel_pred) + self.smooth))
        tsens = ((torch.sum(torch.multiply(skel_true, inputs[:, 1])) + self.smooth)
                 / (torch.sum(skel_true) + self.smooth))
        # tprec = (torch.sum(torch.multiply(skel_pred, skel_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        # tsens = (torch.sum(torch.multiply(skel_pred, skel_true)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice


def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dice(inputs, targets):
    """[function to compute dice loss]

    Args:
        inputs ([float32]): [predicted image]
        targets ([float32]): [ground truth image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((targets * inputs))
    coeff = (2. * intersection + smooth) / (torch.sum(targets) + torch.sum(inputs) + smooth)
    return (1. - coeff)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    outer = F.relu(img - img1)
    for j in range(iter_):
        img1 = soft_erode(img1)
        img2 = soft_open(img1)
        delta = F.relu(img - img2)
        outer = outer + F.relu(delta - outer * delta)
    return img - outer


class BCEGeneralizedDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEGeneralizedDiceLoss, self).__init__()
        self.alpha = alpha
        # self.bce = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = GeneralizedDiceLoss()

    def forward(self, input, target, mask=None):
        if target.dim() == (input.dim() - 1):
            n_classes = input.size()[1]
            if target.dim() >= 3:
                if mask is not None and mask.shape == target.shape:
                    target = expand_as_one_hot(target, C=n_classes)
                    weight = mask[:, None].float()
                    target = target * weight
                    input = input * weight
                else:
                    target = expand_as_one_hot(target, C=n_classes)
            elif input.shape[1:] == target.shape:
                # elif input.shape[1:] == target.shape:
                target = torch.nn.functional.one_hot(target, input.shape[0]).T
                target = target.to(input.dtype)
            else:
                # onehot-class first channel
                order = [i for i in range(target.ndim)]
                target = torch.nn.functional.one_hot(target, n_classes).permute(
                    [0, target.ndim, *order[1:]]).contiguous()
                target = target.to(input.dtype)

        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class SegmentOffsetLoss(nn.Module):
    def __init__(self, alpha=.5, beta=100., **kwargs):
        super(SegmentOffsetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = BCEDiceLoss(.5, .5)
        self.smooth_l1 = SmoothL1Loss()
        self.kwargs = kwargs
        self.seg_mask = kwargs.get('seg_mask', False)

    def forward(self, pred, target):
        pred_seg, pred_offset = pred
        target_seg, target_offset = target
        t_seg, mask = target_seg.split(1, dim=1)
        t_seg_squeeze = t_seg.squeeze(1)
        n_classes = pred_seg.size(1)
        repeats = torch.div(torch.tensor(target_offset.shape), torch.tensor(t_seg.shape), rounding_mode='trunc')
        seg_positive = t_seg > 0

        offset_positive = seg_positive.repeat(*repeats)

        t_seg_onehot = expand_as_one_hot(t_seg_squeeze, C=n_classes)
        eps = 1e-6
        pred_offset_norm = pred_offset / (pred_offset.norm(dim=1, keepdim=True) + eps)
        target_offset_norm = target_offset / (target_offset.norm(dim=1, keepdim=True) + eps)
        offset_dot = torch.sum( pred_offset_norm * target_offset_norm, dim=1, keepdim=True)
        offset_dot = 1 - ( offset_dot ** 2)

        return {
            'seg': self.alpha * self.dice(pred_seg, t_seg_onehot),
            'offset': self.beta * self.smooth_l1(pred_offset[offset_positive], target_offset[offset_positive]),
            'displacement': offset_dot[seg_positive].mean(),
        }



class MeshSegmentOffsetLoss(nn.Module):
    def __init__(self, alpha=.5, beta=100., gamma=1.0, **kwargs):
        super(MeshSegmentOffsetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dice = BCEDiceLoss(.5, .5)
        self.smooth_l1 = SmoothL1Loss()
        self.kwargs = kwargs
        self.seg_mask = kwargs.get('seg_mask', True)


    def forward(self, pred, target):
        pred_seg, pred_offset = pred[0] if len(pred) == 1 else pred
        target_seg, target_offset = target[0] if len(target) == 1 else target
        t_seg, mask = target_seg.split(1, dim=1)
        t_seg_squeeze = t_seg.squeeze(1)
        n_classes = pred_seg.size(1)
        repeats = torch.div(torch.tensor(target_offset.shape), torch.tensor(t_seg.shape), rounding_mode='trunc')
        seg_repeats = torch.div(torch.tensor(pred_seg.shape), torch.tensor(t_seg.shape), rounding_mode='trunc')
        off_positive = t_seg == 1.
        seg_positive = mask > .5
        seg_positive_repats = seg_positive.repeat(*seg_repeats)
        offset_positive = off_positive.repeat(*repeats)

        t_seg_onehot = expand_as_one_hot(t_seg_squeeze, C=n_classes)
        eps = 1e-6
        pred_offset_norm = pred_offset / (pred_offset.norm(dim=1, keepdim=True) + eps)
        target_offset_norm = target_offset / (target_offset.norm(dim=1, keepdim=True) + eps)
        offset_dot = torch.sum( pred_offset_norm * target_offset_norm, dim=1, keepdim=True)
        offset_dot = 1 - ( offset_dot ** 2)

        return {
            'seg': self.alpha * self.dice(pred_seg[seg_positive_repats], t_seg_onehot[seg_positive_repats]),
            # 'seg': self.alpha * self.dice(pred_seg, t_seg_onehot),

            'offset': self.beta * self.smooth_l1(pred_offset[offset_positive], target_offset[offset_positive]),
            'displacement': self.gamma * offset_dot[seg_positive].mean(),
        }


class SegmentOffsetLoss2(nn.Module):
    def __init__(self, param1, param2):
        super(SegmentOffsetLoss2, self).__init__()
        self.seg_offset_loss = SegmentOffsetLoss(**param1)

    def forward(self, preds, targets) -> Dict[str, torch.Tensor]:
        losses = {}
        for i, (y_pred, y_target) in enumerate(zip(preds, targets)):
            prefix = f'{i:02d}_'
            loss = self.seg_offset_loss(y_pred, y_target)
            losses.update({
                prefix + k: v for k, v in loss.items()
            })

        return loss


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, output, target):
        if output.size() == target.size():
            pass
        elif output.ndim == target.ndim + 1:
            # target = expand_as_one_hot(target, output.shape[-1])
            target = F.one_hot(target, output.shape[-1]).to(output.dtype)
        return self.loss(output, target)
    # def _slow_forward(self, *input, **kwargs):

class SegmentClassificationLoss(nn.Module):
    def __init__(self, **config):
        super(SegmentClassificationLoss, self).__init__()
        num_class = config.get('num_class', 33)
        w0 = [1.] * num_class
        w0[0] = 0.5
        w = torch.tensor(w0).cuda()
        # weights = torch.ones([], dtype=torch.float32).cuda()
        self.bce_seg_dice = BCEDiceLoss(.5, .5)
        # self.classification_criterion = nn.CrossEntropyLoss(weight=w)
        self.classification_criterion = BCEWithLogitsLoss(pos_weight=w)
        self.stat_class_weight = config.get('stat_class_weight', 0.01)
        self.pred_class_weight = config.get('pred_class_weight', 1.0)

    def forward(self, preds, targets):
        from trainer import torch_utils
        preds, targets = torch_utils.squeeze((preds, targets))
        pred_mask, pred_classes = preds
        target_mask, target_class = targets
        seg_loss = self.bce_seg_dice(pred_mask, target_mask)
        num_actucal_pred = pred_mask.shape[0]

        # expand_as_one_hot()
        actucal_cls_loss = self.classification_criterion(pred_classes[:num_actucal_pred], target_class[:num_actucal_pred])
        if pred_classes[num_actucal_pred:].size(0) > 0:
            stat_clss_loss = self.classification_criterion(pred_classes[num_actucal_pred:], target_class[num_actucal_pred:])
        else:
            stat_clss_loss = 0.
        class_loss = self.pred_class_weight * actucal_cls_loss + self.stat_class_weight * stat_clss_loss
        # cls_loss = self.class_seg_dice(pred_classes, target_class)
        return {
            'seg_loss': seg_loss,
            'class_loss': class_loss
        }


        # self.


def _create_loss(name, loss_config):
    model_class = get_class(name, modules=[
        'trainer.test.testmodel',
        'trainer.losses',
        'interfaces.pidnetmodel',
    ])
    return create_class(model_class, loss_config)
    # # first as dcitonary
    # try:
    #     return model_class(**loss_config)
    # except Exception as e:
    #     logger.warning(e.args)
    #
    # try:
    #     arg = ml_collections.ConfigDict(loss_config)
    #     return model_class(arg)
    # except Exception as e:
    #     logger.warning(e.args)
    #
    # try:
    #     return model_class()
    # except Exception as e:
    #     raise NotImplementedError(e.args)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config['name']

    return _create_loss(name, loss_config)



