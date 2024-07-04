import copy
from typing import Optional
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from .buildingblocks import SingleConv, DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from trainer.torch_utils import number_of_features_per_level, data_convert, to_numpy, get_class
from trainer import get_model
from trainer import torch_utils
# from torch.utils import get_class
# from teethnet.models.unet3d.utils import get_class, number_of_features_per_level, data_convert, to_numpy


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True, **kwargs):
        super(AbstractUNet, self).__init__()
        if isinstance(basic_module, str):
            basic_module = get_class(basic_module, ['models.model'])
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d, **kwargs)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d, **kwargs)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        # if not self.training and self.final_activation is not None:
        if self.final_activation is not None:

            x = self.final_activation(x)

        return x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=kwargs.pop('basic_module', 'DoubleConv'),
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=True, **kwargs)

class Unet3DClassify(UNet3D):
    def __init__(self, *args, **kwargs):
        super(Unet3DClassify, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.ndim == 6:
            x = torch.squeeze(x, dim=0)
        return super(Unet3DClassify, self).forward(x)


class UNet3DPredictor(UNet3D):
    """
    detecion 모듈로 사용하는 클래스
    """
    def __init__(self, *args, **kwargs):
        super(UNet3DPredictor, self).__init__(*args, **kwargs)
        self.in_channel = -1

    def get_input_channel(self):
        if self.in_channel < 0:
            for params in self.named_parameters():
                name, param = params
                if name.find('conv') >= 0:
                    self.in_channel = param.shape[1]
                    break

        return self.in_channel

    def device(self):
        return next(self.parameters()).device

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 3
        rx = x.reshape([1, *x.shape])
        rx = np.repeat(rx, self.get_input_channel(), axis=0)
        tx = data_convert(rx)

        with torch.set_grad_enabled(self.training):
            res = super(UNet3D, self).forward(tx)

        return to_numpy(res)


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             is3d=True)


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on 
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNetSE3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               final_sigmoid=final_sigmoid,
                                               basic_module=ResNetBlockSE,
                                               f_maps=f_maps,
                                               layer_order=layer_order,
                                               num_groups=num_groups,
                                               num_levels=num_levels,
                                               is_segmentation=is_segmentation,
                                               conv_padding=conv_padding,
                                               is3d=True)


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     is3d=False)



class Unet3DROI(UNet3D):

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(Unet3DROI, self).__init__(in_channels, out_channels, final_sigmoid=final_sigmoid, f_maps=f_maps, layer_order=layer_order,
                 num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation, conv_padding=conv_padding, **kwargs)


    def forward(self, x, y, is_roi):

        x = x + y * is_roi

        return super(Unet3DROI, self).forward(x)

#
#
# def get_model(model_config):
#     model_class = get_class(model_config['name'], modules=[
#         'teethnet.models.unet3d.model',
#         'planning_gan.models.transunet.transunet3d.vit_seg_modeling',
#         'planning_gan.models.transunet.transunet3d.vit_seg_interpolator_modeling',
#         'teethnet.models.pytorchyolo3d.model',
#         'teethnet.models.relationobject.model',
#         'teethnet.models.unetrpp.network_architecture.lung.unetr_pp_lung',
#         'teethnet.models.voxelmorph.networks',
#     ])
#     try:
#         return model_class(**model_config)
#     except TypeError as e:
#         arg = ml_collections.ConfigDict(model_config)
#         return model_class(arg)
#     except Exception as e:
#         raise ValueError(e.args)


class Unet3DSegOffset(nn.Module):
    def __init__(self, in_channels, final_sigmoid, basic_module=ResNetBlock, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, is3d=True, **kwargs):
        super(Unet3DSegOffset, self).__init__()

        segment_ch = kwargs.get('segment_ch')
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if 'g' in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"
        if isinstance(basic_module, str):
            basic_module = get_class(basic_module, ['models.model'])


        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d, pool_type=kwargs.get('pool_type', 'max'),
                                        se_module=kwargs.get('se_module', 'scse'))

        # create decoder path
        self.decoders1 = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d,  pool_type=kwargs.get('pool_type', 'max'),
                                        se_module=kwargs.get('se_module', 'scse'))
        head_kernel = kwargs.get('head_kernel', 3)
        self.double_decoder = kwargs.get('double_decoder', False)
        self.decoders2 = copy.deepcopy(self.decoders1) if self.double_decoder else None
        padding = head_kernel // 2
        if is3d:
            ch = f_maps[0]
            self.shared = basic_module(ch, ch,
                                       kernel_size=3,
                                       encoder=False,
                                       order=layer_order,
                                       num_groups=num_groups,
                                       padding=conv_padding,
                                       is3d=is3d)
            self.segment_head = nn.Conv3d(f_maps[0], segment_ch, head_kernel, padding=padding)

            self.offset_head = nn.Conv3d(f_maps[0], 3, head_kernel, padding=padding)
        else:
            pass

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        if self.double_decoder:
            x1 = self.forward_decoder(x, encoders_features, self.decoders1)
            x2 = self.forward_decoder(x, encoders_features, self.decoders2)
            # shared = self.shared(x)
            seg, offset = self.segment_head(x1), self.offset_head(x2)
        else:
            x = self.forward_decoder(x, encoders_features, self.decoders1)
            shared = self.shared(x)
            seg, offset = self.segment_head(shared), self.offset_head(shared)
        # x2 = self.forward_decoder(x, encoders_features, self.decoders2, self.final_conv2)
        if self.final_activation:
            seg = self.final_activation(seg)
        # output target 갯수만큼
        return ((seg, offset), )

    def forward_decoder(self, x, encoders_features, decoders):
        for decoder, encoder_features in zip(decoders, encoders_features):
            x = decoder(encoder_features, x)

        return x


class Unet3DCouple(nn.Module):
    def __init__(self, args):
        super(Unet3DCouple, self).__init__()
        # self.model1 = UNet3D(**args.model1)
        # self.model2 = UNet3D(**args.model2)
        # ml_collections.ConfigDict().to_dict()
        self.model1 = get_model(args.model1) #(**args.model1.to_dict())
        # self.model2 = get_model(args.model2) #(**args.model1.to_dict())

    def forward(self, x):
        y1, y2 = self.forward_two_model(x)
        # y1 = self.model1(x)
        # y2 = self.model2(x)
        y1_seg, y1_offset = y1
        # y2_seg, y2_offset = y2
        # return (y1_seg, y1_offset), (y2_seg, y2_offset)
        return (y1_seg, y1_offset), (y2_seg, y2_offset)

    def forward_two_model(self, x):
        y1 = self.model1(x)
        y2 = self.model2(x)
        return y1, y2



class Decoder(nn.Module):
    def __init__(self, model_config, model_key, hidden_ch):
        super(Decoder, self).__init__()
        decoder_params = [
            'f_maps', 'basic_module', 'conv_kernel_size', 'conv_padding', 'layer_order', 'num_groups', 'is3d'
        ]
        # model_config =
        config = model_config[model_key]
        config_params = {k: config[k] for k in decoder_params}
        basic_module = config_params['basic_module']
        self.decoders1 = create_decoders(**config_params)

        if isinstance(basic_module, str):
            basic_module = globals()[basic_module]

        if hidden_ch > 0:
            encode_ch = config_params['f_maps'][-1]
            # hidden_size = model_config['hidden_size']
            self.conv_more = basic_module(
                hidden_ch,
                encode_ch,
                encoder=False,
                kernel_size=config_params['conv_kernel_size'],
                order=config_params['layer_order'],
                num_groups=config_params['num_groups'],
                padding=config_params['conv_padding'],
                is3d=config_params['is3d']
            )
        else:
            self.conv_more = None
            # hidden_size,
            # encode_ch,
            # kernel_size=config['conv_kernel_size'],
            # padding=1,
            # use_batchnorm=True,

    def forward(self, x, encoder_features):
        assert len(encoder_features) == len(self.decoders1)
        if self.conv_more:
            x = self.conv_more(x)
        for enc_feat, decoder in zip(encoder_features, self.decoders1):
            x = decoder(enc_feat, x)
        return x


class ROIBackbone(nn.Module):
    def __init__(self, config, key='backbone_model'):
        super(ROIBackbone, self).__init__()
        model_config = config[key]
        self.encoders = create_encoders(**model_config) if config.get('backbone', False) else None
        self.return_intermediate_feature = model_config.get('return_intermediate_feature', False)
        self.start_encode_feature = model_config.get('start_encode_feature', 1)
        self.in_channels = model_config['in_channels']
        fmaps = config[key]['f_maps']
        last = fmaps[-1]
        # *final grid ** 3(
        down_size = list(config['pool_shape'])
        for _ in range(len(fmaps) - 1):
            down_size = [v // 2 for v in down_size]
        self.bottom_size = down_size
        feat_size = np.prod(down_size)
        hidden_size = config['hidden_size']
        if hidden_size > 0:
            assert hidden_size % feat_size == 0, f'must to be feat size multiplication: {hidden_size} % {feat_size} != 0'
            assert hidden_size // feat_size > 0, f'must to be greater than feat size'
            out_ch = hidden_size // feat_size
            self.hidden_ch = out_ch
            self.hidden_conv = nn.Conv3d(last, out_ch, kernel_size=1)
        else:
            self.hidden_ch = -1
            self.hidden_conv = nn.Identity()


    def forward(self, x):
        features = []
        for module in self.encoders:
            # print(x.shape)
            x = module(x)
            features.insert(0, x)
        if self.return_intermediate_feature:
            return self.hidden_conv(x), features[self.start_encode_feature:]
        else:
            return self.hidden_conv(x)



class ROIAttentionSegmentClassifier(nn.Module):
    def __init__(self, config):
        super(ROIAttentionSegmentClassifier, self).__init__()

        self.split_backbone_segment_classification = config.get('used_multiple_backbone', False)
        # copy.deepcopy(self.backbone1)
        self.backbone1 = ROIBackbone(config, key='backbone_model1')
        hidden_conv_off = {**config, 'key': 'backbone_model1', 'hidden_size': -1}
        seg_backbone_config_key = 'backbone_seg' if ('backbone_seg' in config) else 'backbone_model1'

        # hidden_conv_off['backbone_model1']['in_channes'] = config.get('segnebt_backbone_in_channels') or config.get('in_channels')
        self.backbone1_seg = ROIBackbone(hidden_conv_off,
                                         key=seg_backbone_config_key) if self.split_backbone_segment_classification else None
        self.backbone2 = ROIBackbone(config, key='backbone_model2') if config[
            'used_multiple_backbone'] else self.backbone1
        self.used_segment_transformer = config.get('used_segment_transformer', True)  # 기존 default option
        self.classified_feat_used_segment = config.get('classified_feat_used_segment', False)

        self.stat_attention = config.get('stat_attention', True)
        # copy()
        # self.segment_use
        assert self.backbone1.return_intermediate_feature, 'we need intermediate features'
        assert self.backbone2.return_intermediate_feature, 'we need intermediate features'
        # self.backbone1 = copy(self.backbone)
        self.pool_shape = tuple(config.get('pool_shape'))

        # from planning_gan.models.transunet.transunet3d import vit_seg_modeling

        # self.encoder = vit_seg_modeling.Encoder(config, True)
        self.encoder = TransformerEncoder(config)
        # self.encoder1 = TransformerEncoder(config) if config[]
        # self.de

        self.dim_g = config.transformer.geo_feature_dim
        self.wave_len = config.transformer.wave_len
        ch = config['hidden_size']
        n_classes = config['n_classes']
        self.temperature = config.get('temperature', 100)
        self.hidden_dim = ch



        position_embedding_func = config.get('position_embedding_func', 'position_embedding_sin3d')
        self.position_embedding = globals()[position_embedding_func]

        self.fc1 = nn.Linear(ch, ch)
        self.fc2 = nn.Linear(ch, n_classes)

        self.num_geo_dim = config.get('num_geo_dim', 3)

        f_maps = config['backbone_model1']['f_maps']
        final_sigmoid = config['backbone_model1'].get('final_sigmoid', True)
        segment_decoder_in_ch = self.backbone1_seg.hidden_ch if self.used_segment_transformer else -1
        self.decoder = Decoder(config, 'backbone_model1', segment_decoder_in_ch)
        self.final_segment_conv = self.final_conv = nn.Conv3d(f_maps[0], 2, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

        final_classification_activation = config.get('final_classification_activation', 'softmax')
        if final_classification_activation == 'softmax':
            final_classification_func = nn.Softmax(dim=1)
        elif final_classification_activation == 'identity':
            final_classification_func = nn.Identity()
        else:
            raise ValueError(final_classification_activation)
        self.final_classification_activation = final_classification_func

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # comments
        for sm in self.modules():
            weight_init_xavier_uniform(sm)
            # if isinstance(sm, nn.Conv3d):
            #     torch.nn.init.normal_(
            #         sm.weight.data,
            #         mean=0.0,
            #         std=0.02
            #         )


    def forward_backbone(self, inputs, backbone:ROIBackbone):
        if backbone:
            in_ch = backbone.in_channels
            num = inputs.shape[0]
            # k3 = torch.pow(torch.tensor(features.shape[-1]), 1 / 3).to(torch.int32)
            # assert k3 ** 3 == features.shape[-1], 'we assume all shape size is same'
            # features = features.reshape((num,) + (1,) + (k3,) * 3)
            shape = self.pool_shape
            features = inputs.reshape((num,) + (in_ch,) + shape)
            # backbone.re
            x, features = backbone(features)

            if backbone.hidden_ch > 0:
               return x.flatten(1), features
            else:
                return x, features
        else:
            return inputs

    def restore_backbone_shape(self, inputs, backbone:ROIBackbone):
        in_ch = backbone.in_channels
        num = inputs.shape[0]
        # k3 = torch.pow(torch.tensor(features.shape[-1]), 1 / 3).to(torch.int32)
        # assert k3 ** 3 == features.shape[-1], 'we assume all shape size is same'
        # features = features.reshape((num,) + (1,) + (k3,) * 3)
        shape = self.pool_shape
        return inputs.reshape((num,) + (in_ch,) + shape)
        # backbone.re
        # x, features = backbone(features)

        # return self.backbone(features) if self.backbone else features
    def squeeze_image(self, inputs1, inputs2):
        inputs1, inputs2 = torch_utils.squeeze((inputs1, inputs2))
        return inputs1, inputs2

    def encoding(self, inputs1, inputs2):
        inputs1, inputs2 = self.squeeze_image(inputs1, inputs2)
            # from tools import to
        # torch_utils.squeeze()
        (image1, rois1), (image2, rois2) = inputs1, inputs2
        # assert len(inputs) == 2 and
        # assert roi_feats.ndim == 2 or (roi_feats.ndim == 3 and roi_feats.shape[0] == 1)
        # assert roi_feats.ndim ==
        # num_queries = image1.size(0)


            # (roi_feats1, features1)

        if self.split_backbone_segment_classification:
            last_feat, features1_seg = self.forward_backbone(image1, self.backbone1_seg) # if self.split_backbone_segment_classification else \
        else:
            roi_feats1, features1 = self.forward_backbone(image1, self.backbone1)
            roi_feats2, features2 = self.forward_backbone(image2, self.backbone2)
            last_feat, features1_seg = (features1[0], features1[1:]) if self.backbone1.start_encode_feature == 0 else (None, features1)



        if not self.classified_feat_used_segment:
            hs = self._encoding_classify(inputs1, inputs2)

        else:
            hs = None

        return last_feat, features1_seg, hs

    def _encoding_classify(self, inputs1, inputs2):
        inputs1, inputs2 = self.squeeze_image(inputs1, inputs2)

        (image1, rois1), (image2, rois2) = inputs1, inputs2

        roi_feats1, features1 = self.forward_backbone(image1, self.backbone1)
        roi_feats2, features2 = self.forward_backbone(image2, self.backbone2)
            # (roi_feats1, features1)

        # pose_embedding = self.position_embedding(rois, self.hidden_dim)
        pose_embedding1 = self.position_embedding(rois1[..., :self.num_geo_dim], self.hidden_dim,
                                                  temperature=self.temperature)

        pose_embedding2 = self.position_embedding(rois2[..., :self.num_geo_dim], self.hidden_dim,
                                                  temperature=self.temperature)

        if self.stat_attention:
            roi_feats = torch.cat([roi_feats1, roi_feats2], dim=0)
            pose_emb = torch.cat([pose_embedding1, pose_embedding2], dim=0)
        else:
            pose_emb = pose_embedding1
            roi_feats = roi_feats1
            # roi_feats1 = roi

        hs = self.encoder(roi_feats[:, None], pos=pose_emb[:, None])
        hs = torch.squeeze(hs)
        return hs


    def forward(self, inputs1, inputs2):
        # inputs1, inputs2 = self.squeeze_image(inputs1, inputs2)
        #     # from tools import to
        # # torch_utils.squeeze()
        # (image1, rois1), (image2, rois2) = inputs1, inputs2
        # # assert len(inputs) == 2 and
        # # assert roi_feats.ndim == 2 or (roi_feats.ndim == 3 and roi_feats.shape[0] == 1)
        # # assert roi_feats.ndim ==
        # num_queries = image1.size(0)
        #
        # roi_feats1, features1 = self.forward_backbone(image1, self.backbone1)
        # roi_feats2, features2 = self.forward_backbone(image2, self.backbone2)
        #
        # last_feat, features1 = (features1[0], features1[1:]) if self.backbone1.start_encode_feature == 0 else (None, features1)
        # # ndim = roi_feats.ndim
        # # if ndim == 3:
        # #     roi_feats, rois = roi_feats.squeeze(), rois.squeeze()
        #
        # # pose_embedding = self.position_embedding(rois, self.hidden_dim)
        # pose_embedding1 = self.position_embedding(rois1[..., :self.num_geo_dim], self.hidden_dim,
        #                                          temperature=self.temperature)
        #

        seg_image, seg_features, class_feat = self.encoding(inputs1, inputs2)

        if self.used_segment_transformer:
            # if sdf
            seg_x = self.decoder(seg_image, seg_features)
        else:
            # assert last_feat is not None, 'we need bakbone last feature'
            # feats = self.restore_backbone_shape(la, self.backbone1)
            seg_x = self.decoder(seg_image, seg_features)
        seg_x = self.final_segment_conv(seg_x)
        seg_x = self.final_activation(seg_x)
        # hs = hs[None]
        # hs = hs.squeeze()
        if self.classified_feat_used_segment:
            pred_seg_x = torch.argmax(seg_x, dim=1, keepdim=True).to(dtype=seg_x.dtype)
            inputs1_seg = (pred_seg_x, inputs1[1])
            class_feat = self._encoding_classify(inputs1_seg, inputs2)
        else:
            pass

        x = self.fc1(class_feat)
        x = self.fc2(x)
        # if self.final_loss
        pred_class = self.final_classification_activation(x)
        # outputs = outputs.permute(1, 0).contiguous()[None]
        return seg_x, pred_class


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, (torch.nn.Conv2d, torch.nn.Conv3d)):
        torch.nn.init.xavier_uniform_(submodule.weight)
        if submodule.bias is not None:
            submodule.bias.data.fill_(0.01)
        # print('weights-init')
    elif isinstance(submodule, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
        submodule.weight.data.fill_(1.0)
        if submodule.biasis is not None:
            submodule.bias.data.zero_()
        # print('weight init')
    elif isinstance(submodule, nn.Linear):
        nn.init.kaiming_uniform_(submodule.weight.data)
        if submodule.bias is not None:
            nn.init.constant_(submodule.bias.data, 0)
        # print('weights-init')

    else:
        pass


# import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config['hidden_size']
        nhead = config['transformer']['num_heads']
        dim_feedforward = config['transformer'].get('dim_feedforward') or d_model
        dropout = 0.1
        activation = 'relu'
        normalize_before = config.get('normalize_before', False)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config['hidden_size']
        num_layers = config['transformer'].get('num_layers', 3)
        normalize_before = config['transformer'].get('normalize_before')
        encoder_layer = TransformerEncoderLayer(config)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = encoder_norm

    def forward(self,
                src,
                mask: Optional[Tensor]=None,
                src_key_padding_mask: Optional[Tensor]=None,
                pos: Optional[Tensor]=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}")



def position_embedding_sin3d_xyzxyz(pose: torch.Tensor,
                              num_pos_feats, temperature=100, eps=1e-6):
    assert pose.shape[-1] >= 6
    ctr, size = pose[..., :3], pose[..., 3:]
    bmin, bmax = ctr - size / 2, ctr + size / 2
    aabb = torch.cat([bmin, bmax], dim=-1)
    return position_embedding_sin3d(aabb, num_pos_feats, temperature, eps)




def deltas_centers(centr1, length1, epsilon, centr2=None):
    centr2 = centr1 if centr2 is None else centr2
    # length2 = length2 or length1.view(1, -1)
    delta_x = centr1 - centr2.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / (length1 + epsilon)), min=epsilon)
    delta_x = torch.log(delta_x)
    return delta_x


def deltas_lengths(length1, epsilon, length2=None):
    length1_abs = length1.clamp(min=epsilon)
    length2_abs = length2.clamp(min=epsilon) if length2 is not None else length1_abs

    return torch.log(length1_abs / (length2_abs.view(1, -1) + epsilon))


def deltas_angles(angles1, epsilon, angles2=None):
    angles2 = angles2 if angles2 is not None else angles1
    angles1_abs = (angles1 + torch.pi).abs()
    angles2_abs = (angles2 + torch.pi).abs()
    # return (angles1 - angles2.view(1, -1)) / torch.pi
    # return (angles1 - angles2.view(1, -1)) / torch.pi

    return torch.log(angles1_abs / (angles2_abs.view(1, -1) + epsilon))

def position_embedding_3d(f_g, dim_g=64, wave_len=1000, epsilon=1e-5, f_g2=None):
    assert f_g.shape[-1] == 6
    assert dim_g % 12 == 0
    assert f_g2 is None, 'not implemented'
    z_min, y_min, x_min, z_max, y_max, x_max = torch.chunk(f_g, 6, dim=1)
    # epsilon
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    cz = (z_min + z_max) * 0.5
    h = (y_max - y_min)
    w = (x_max - x_min)
    d = (z_max - z_min)

    centers = [cz, cy, cx]
    lengths = [d, h, w]
    delta_z, delta_y, delta_x = [deltas_centers(center, length, epsilon) for center, length in zip(centers, lengths)]
    delta_d, delta_h, delta_w = [deltas_lengths(length, epsilon) for length in lengths]

    size = delta_d.size()

    delta_z, delta_y, delta_x, delta_d, delta_h, delta_w = \
        [val[..., None] for val in [delta_z, delta_y, delta_x, delta_d, delta_h, delta_w]]

    position_mat = torch.cat([delta_z, delta_y, delta_x, delta_d, delta_h, delta_w], -1)

    feat_range = torch.arange(dim_g / 12, device=f_g.device)
    dim_mat = feat_range / (dim_g / 12)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 6, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    return embedding


def position_embedding_sin3d(pose: torch.Tensor,
                              num_pos_feats, temperature=100, eps=1e-6):
    """
    [res] DETR pose_embedding
    Parameters
    ----------
    tensor :
    num_pos_feats :
    normalize :
    scale :
    temperature :
    eps :

    Returns
    -------

    """

    assert num_pos_feats % pose.size(-1) == 0
    num_pos_feats = int(num_pos_feats // pose.size(-1)) * 2
    dtype = pose.dtype
    device = pose.device
    # weight = torch.ones([pose.size(-1)], dtype=dtype, device=device)
    # weight[-3:] = 2 * torch.pi
    freq = temperature ** (torch.arange(0, num_pos_feats, 2, dtype=dtype, device=device) / num_pos_feats)
    inv_freq = 1. / freq

    embeds = []
    splits = torch.split(pose, 1, dim=-1)
    for emb in splits:
        emb = emb * inv_freq
        emb = torch.stack((emb[..., 0::2].sin(), emb[..., 1::2].cos()), dim=-1)
        embeds.append(emb)
    return torch.cat(embeds, dim=-1).flatten(1)  # .permute(0, 2, 1).contiguous()