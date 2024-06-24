# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import math
import copy

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv3d, LayerNorm
import torch.nn.functional as F

from .buildingblocks import create_encoders, create_decoders
from .vit_seg_modeling_resnet_skip import ResNetV2
from .vit_seg_modeling import DecoderCup, SegmentationHead, Encoder, final_active_function, Conv3dReLU
logger = logging.getLogger(__name__)


class DecoderInterpolator(DecoderCup):
    def __init__(self, config):
        super(DecoderInterpolator, self).__init__(config)
        self.transformer_grid = config.patches.grid

        used_conv_more2 = config.get('used_conv_more2')
        if used_conv_more2:
            self.conv_more2 = Conv3dReLU(
                config.hidden_size,
                self.config.decoder_channels[0],
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
        else:
            self.conv_more2 = None

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # d = h = w = round(np.power(n_patch, 1/3))# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # d = h = w = torch.round(torch.pow(torch.tensor(n_patch), 1 / 3)).to(torch.int64)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # (1, 196, 768) -? (1, 14x14, 768) // transpose -> (1, 768, 14x14) . 14 = 224 / 16
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, *self.transformer_grid)
        # 보간
        # target_shape = torch.tensor(features[-1].shape[2:] * 2) // 16
        # x2 = self.conv_more2(x)
        target_shape = (torch.tensor(features[-1].shape[2:]) * 2) / 16
        # x = F.interpolate(x, tuple(target_shape), mode='trilinear')
        x = F.interpolate(x, list(target_shape.int()), mode='trilinear', align_corners=True)
        #
        x = self.conv_more(x)
        #
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            # if i == 0:
            #     skip  = skip + x2
            # print('x', x.shape, '\t skip', skip.shape if skip else 'None', next(decoder_block.parameters()).shape)
            x = decoder_block(x, skip=skip)
            # print('decoder', x.shape)
        return x



class DecoderInterpolatorAttenInterpolation(DecoderCup):
    def __init__(self, config):
        super(DecoderInterpolatorAttenInterpolation, self).__init__(config)
        self.transformer_grid = config.patches.grid

        self.conv_more2 = Conv3dReLU(
            config.hidden_size,
            self.config.decoder_channels[0],
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        self.num_atten_conv = 3


        self.conv_more_lists = nn.ModuleList()
        # 첫번째꺼는 직접
        for i in range(self.num_atten_conv):
            self.conv_more_lists.append(
                Conv3dReLU(
                    config.hidden_size,
                    config.decoder_channels[i],
                    kernel_size=3,
                    padding=1,
                    use_batchnorm=True,
                )
            )


    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # d = h = w = round(np.power(n_patch, 1/3))# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # d = h = w = torch.round(torch.pow(torch.tensor(n_patch), 1 / 3)).to(torch.int64)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # (1, 196, 768) -? (1, 14x14, 768) // transpose -> (1, 768, 14x14) . 14 = 224 / 16
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, *self.transformer_grid)
        atten_x = x
        # 보간
        # target_shape = torch.tensor(features[-1].shape[2:] * 2) // 16
        # x2 = self.conv_more2(x)
        target_shape = (torch.tensor(features[-1].shape[2:]) * 2) / 16
        # x = F.interpolate(x, tuple(target_shape), mode='trilinear')
        x = F.interpolate(x, list(target_shape.int()), mode='trilinear', align_corners=True)
        #
        x = self.conv_more(x)
        #
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            if 0 < i < self.num_atten_conv:
                # conv_atten_x = self.conv_more_lists[0](atten_x)
                interp_atten_x = F.interpolate(atten_x, x.size()[2:], mode='trilinear', align_corners=True)
                x = x + self.conv_more_lists[i-1](interp_atten_x)
                # F.interpolate(atten_x, )

            # if i == 0:
            #     skip  = skip + x2
            # print('x', x.shape, '\t skip', skip.shape if skip else 'None', next(decoder_block.parameters()).shape)
            x = decoder_block(x, skip=skip)
            # print('decoder', x.shape)
        return x


from .vit_seg_modeling import Conv3dReLU, DecoderBlock

class DecoderInterpolatorV3(nn.Module):
    def __init__(self, config):
        super(DecoderInterpolatorV3, self).__init__()
        self.transformer_grid = config.patches.grid

        self.config = config
        head_channels = 120
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        n_skip = len(self.config.skip_channels)
        skip_channels = self.config.skip_channels

        # if self.config.n_skip != 0:
        #     pass
        #     skip_channels = self.config.skip_channels
        #     for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
        #         skip_channels[3-i]=0
        #
        # else:
        #     skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # d = h = w = round(np.power(n_patch, 1/3))# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # d = h = w = torch.round(torch.pow(torch.tensor(n_patch), 1 / 3)).to(torch.int64)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # (1, 196, 768) -? (1, 14x14, 768) // transpose -> (1, 768, 14x14) . 14 = 224 / 16
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, *self.transformer_grid)
        # 보간
        # target_shape = torch.tensor(features[-1].shape[2:] * 2) // 16
        target_shape = (torch.tensor(features[-1].shape[2:])) / 16
        # x = F.interpolate(x, tuple(target_shape), mode='trilinear')
        x = F.interpolate(x, list(target_shape.int()), mode='trilinear', align_corners=True)
        #
        x = self.conv_more(x)
        #
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i]
                # skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            # print('x', x.shape, '\t skip', skip.shape if skip else 'None', next(decoder_block.parameters()).shape)
            x = decoder_block(x, skip=skip)
            # print('decoder', x.shape)
        return x


from collections import OrderedDict
from .vit_seg_modeling_resnet_skip import StdConv3d, PreActBottleneck


class ResNetV3(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, in_channels=3, channel_factor=4):
        super(ResNetV3, self).__init__()

        block_factor = channel_factor
        width = int(block_factor * width_factor)
        self.width = width
        self.block_units = block_units
        blocks = []
        for num in range(len(block_units)):
            power = 2 ** num
            cin = width * ( 2 ** (num - 1) ) if num > 0 else in_channels
            stride = 2 if  num > 0 else 1
            blocks.append(
                (f'block{num+1}', nn.Sequential(OrderedDict(
                    [('unit1', PreActBottleneck(cin=cin, cout=width * power, cmid=width, stride=stride,
                                                num_groupnorm=block_factor))] +
                    [(f'unit{i:d}',
                      PreActBottleneck(cin=width * power, cout=width * power, cmid=width, num_groupnorm=block_factor)) for i in
                     range(2, block_units[num] + 1)],
                )))
            )
        self.body = nn.Sequential(OrderedDict(blocks))

        self.out_channel = width * ( 2 ** (len(block_units) - 1) )

    def encoder_skip_channels(self):
        enc_channels = []
        for num in range(len(self.block_units)):
            power = 2 ** (num )
            enc_channels.append(self.width * power)
        return enc_channels
        # power = 2 ** (num + 1)

    def forward(self, x):
        features = []
        b, c, in_size, _, _ = x.size()
        # x = self.root(x)
        # features.append(x)
        # x = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            # if x.size()[2] != right_size:
            #     pad = right_size - x.size()[2]
            #     assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            #     feat = torch.zeros((b, x.size()[1], right_size, right_size, right_size), device=x.device)
            #     feat[:, :, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]] = x[:]
            # else:
            feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class EmbeddingsInterpolator(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(EmbeddingsInterpolator, self).__init__()
        self.hybrid = None
        self.config = config
        # img_size = _triple(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            self.transformer_grid = tuple(grid_size)
            n_patches = 1
            for grid in grid_size:
                n_patches *= grid
            patch_size = 1
            self.hybrid = True
        else:
            raise NotImplementedError
            # patch_size = _triple(config.patches["size"])
            # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.hybrid_model_name = config.get('hybrid_model_name', 'ResNetV2')
        if self.hybrid:
            if self.hybrid_model_name == 'ResNetV2':
                self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,
                                             in_channels=config.in_channels)
                in_channels = self.hybrid_model.out_channel
            elif self.hybrid_model_name == 'ResNetV3':
                self.hybrid_model = ResNetV3(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,
                                             in_channels=config.in_channels, channel_factor=config.resnet.channel_factor)
                in_channels = self.hybrid_model.out_channel
            else:
                try:
                    self.hybrid_model = globals()[config.backbone.encoder](config.backbone)
                    in_channels = config.backbone.f_maps[-1]
                except Exception as e:
                    logger.error(e.args)
                    raise ModuleNotFoundError(self.hybrid_model_name)


        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
                # (1, 14*14, 768) = (1, 196, 768)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


        self.learnable_pose_embeddings = self.config.get('learnable_pose_embeddings', True)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # (1, 1024, 14, 14), patch size
        # (1, 768, 14, 14) , patch embedding = conv3d. 1024 ->768, vit 전에 linear-projection
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # 보간 ()
        #
        # shape = (4, 4, 4)
        x = F.interpolate(x, self.transformer_grid, mode='trilinear', align_corners=True)

        if self.learnable_pose_embeddings:
            # (1, 768, 196) = (1, 768, 14*14)
            x = x.flatten(2)


            # (1, 196, 768)
            x = x.transpose(-1, -2)  # (B, n_patches, hidden)

            # (1, 196, 768) + (1, 196, 768)
            embeddings = x + self.position_embeddings
        else:
            x = x + position_embedding_sing3d(x, self.config.hidden_size // 3)
            x = x.flatten(2)
            embeddings = x.transpose(-1, -2)

        embeddings = self.dropout(embeddings)
        # features ( ([1, 512, 28, 28]), ([1, 256, 56, 56]), [1, 64, 112, 112]
        return embeddings, features


class TransformerInterpolator(nn.Module):
    def __init__(self, config, img_size, vis):
        super(TransformerInterpolator, self).__init__()
        self.embeddings = EmbeddingsInterpolator(config, img_size=img_size, in_channels=config.in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        ##
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        ##
        return encoded, attn_weights, features




class EmbeddingsInterpolatorAtten2(EmbeddingsInterpolator):
    def __init__(self, *args, **kwargs):
        super(EmbeddingsInterpolatorAtten2, self).__init__(*args, **kwargs)


        n_patches = self.position_embeddings.shape[1]

        n_patches0 = round(((n_patches ** (1 / 3)) * .5) ** 3)
        #

        self.patch_embeddings0 = Conv3d(in_channels=self.patch_embeddings.weight.shape[1] // 2,
                                        out_channels=self.config.hidden_size,
                                        kernel_size=1,
                                        stride=1)

                # (1, 14*14, 768) = (1, 196, 768)
        self.position_embeddings0 = nn.Parameter(torch.zeros(1, n_patches0, self.config.hidden_size))

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # (1, 1024, 14, 14), patch size
        # (1, 768, 14, 14) , patch embedding = conv3d. 1024 ->768, vit 전에 linear-projection

        if self.learnable_pose_embeddings:

            x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

            # (1, 768, 196) = (1, 768, 14*14)
            x = x.flatten(2)

            # (1, 196, 768)
            x = x.transpose(-1, -2)  # (B, n_patches, hidden)

            # (1, 196, 768) + (1, 196, 768)
            embeddings = x + self.position_embeddings0


            #################


            # TODO:구조가 좀 이상하다. 기존 모델 구조를 어거지로 맞추다보니
            x1 = features[0]
            x1 = self.patch_embeddings0(x1)

            embeddings1 = x1.flatten(2).transpose(-1, -2) + self.position_embeddings
            x1 = embeddings1.transpose(-1, -2).reshape([x1.shape[0], -1, *x1.shape[2:]]).contiguous()
            # replaize
            features[0] = x1


        else:
            x = x + position_embedding_sing3d(x, self.config.hidden_size // 3)
            x = x.flatten(2)
            embeddings = x.transpose(-1, -2)

        embeddings = self.dropout(embeddings)
        # features ( ([1, 512, 28, 28]), ([1, 256, 56, 56]), [1, 64, 112, 112]
        return embeddings, features



class TransformerInterpolatorAtten2(nn.Module):
    def __init__(self, config, img_size, vis):
        super(TransformerInterpolatorAtten2, self).__init__()
        self.embeddings = EmbeddingsInterpolatorAtten2(config, img_size=img_size, in_channels=config.in_channels)
        self.encoder = Encoder(config, vis)
        self.encoder2 = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)

        feat_0 = features[0]
        embedding_output2 = feat_0.flatten(2).transpose(-1, -2)
        # (1, 196, 768)
        encoded2, attn_weights2 = self.encoder2(embedding_output2)

        B, n_patch, hidden = encoded2.size()
        features[0] = encoded2.permute(0, 2, 1).contiguous().view(B, hidden, *[round(n_patch ** (1/3)),]*3) # (B, n_patch, hidden)

        ##
        return encoded, attn_weights, features



class DecoderInterpolatorAtten2(DecoderCup):
    def __init__(self, config):
        super(DecoderInterpolatorAtten2, self).__init__(config)
        self.transformer_grid = config.patches.grid

        self.conv_more2 = Conv3dReLU(
            config.hidden_size,
            self.config.skip_channels[0],
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # d = h = w = round(np.power(n_patch, 1/3))# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # d = h = w = torch.round(torch.pow(torch.tensor(n_patch), 1 / 3)).to(torch.int64)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # (1, 196, 768) -? (1, 14x14, 768) // transpose -> (1, 768, 14x14) . 14 = 224 / 16
        # x = hidden_states.permute(0, 2, 1)
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, *[round(n_patch ** (1/3)),]*3)
        # # 보간

        #
        x = self.conv_more(x)
        features[0] = self.conv_more2(features[0])


        # features[0]
        #
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            # if i == 0:
            #     skip  = skip + x2
            # print('x', x.shape, '\t skip', skip.shape if skip else 'None', next(decoder_block.parameters()).shape)
            x = decoder_block(x, skip=skip)
            # print('decoder', x.shape)
        return x


class VisionTransformerInterpolatorV2(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        super(VisionTransformerInterpolatorV2, self).__init__()
        self.num_classes = config.n_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        # self.transformer = TransformerInterpolator(config, config.img_size, vis)
        self.transformer = globals()[config.get('transformer_name', 'TransformerInterpolator')](config, config.img_size, vis)

        # hybrid_model_name = config.hybrid_model_name
        # if
        decoder_name = config.get('backbone', {}).get('decoder', 'DecoderInterpolator')
        try:
            self.decoder = globals()[decoder_name](config)  # DecoderInterpolator(config)
            decoder_ch = config['decoder_channels'][-1]
        except Exception as e:
            logger.warning(e.args)
            # transformer info 파라미터 copy
            config.backbone.patches = config.patches
            config.backbone.hidden_size = config.hidden_size
            self.decoder = globals()[decoder_name](config.backbone)
            decoder_ch = config.backbone.f_maps[0]

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_ch,
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.final_activation = final_active_function[getattr(config, 'final_activation', 'tanh')]
        # torch.n
        self.config = config

    def forward(self, x):
        # torch.Size([1, 512, 28, 28]), torch.Size([1, 256, 56, 56]), torch.Size([1, 64, 112, 112])]
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # 보간법
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = self.final_activation(logits)
        # logits = torch.tanh(logits)
        return logits


from .vit_seg_modeling import Mlp, Attention


class AttentionQKV(nn.Module):
    def __init__(self, config, vis):
        super(AttentionQKV, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_query, hidden_state_key_value):
        # (1, 196, 768)
        # simliarity(q,k) = Q*k_t/ scaling
        # num_head = 12
        # sacling = sqrt(attention_head_size), attention_head_size 64 = hidden_size / num_head = 768 / 12
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_state_key_value)
        mixed_value_layer = self.value(hidden_state_key_value)

        # (1, num_head, 196, hidden / num_head) = (1, 12, 196, 64)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # (1, 12, 196, 196) = matrix multiplication = (1, 12, 196, 4) X ( 1, 12, 64, 196)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attension_scores = aka similarity = (q * k_t) / scaling ...  scaling = sqrT(attention_head_size)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # weights = softmax(q*k/scaling)
        attention_probs = self.softmax(attention_scores)
        # ( 1, 12, 196, 196) N = 196 = 14 * 14 patch. si
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # (1, 12, 196, 64) = (1, 12, 196, 196) x (1, 12, 196, 64)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (1, 196, 12, 64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # (1, 196, 768)
        context_layer = context_layer.view(*new_context_layer_shape)
        # linear - fully connected
        # (1, 196, 768)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class BlockQKV(nn.Module):
    def __init__(self, config, vis):
        super(BlockQKV, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = AttentionQKV(config, vis)

    def forward(self, query, key_value):

        h = query

        q_x = self.attention_norm(query)
        kv_x = self.attention_norm(key_value)

        x, weights = self.attn(q_x, kv_x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class EncoderQKV(nn.Module):
    def __init__(self, config, vis):
        super(EncoderQKV, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = BlockQKV(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states_q, hidden_states_kv):
        attn_weights = []
        for i, layer_block in enumerate(self.layer):
            if i < len(self.layer) // 2:
                # self attention
                hidden_states_q, weights = layer_block(hidden_states_q, hidden_states_kv)
            else:
                hidden_states_q, weights = layer_block(hidden_states_q, hidden_states_q)

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states_q)

        return encoded, attn_weights


class EmbeddingsROIInterpolator(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(EmbeddingsROIInterpolator, self).__init__()
        self.hybrid = None
        self.config = config

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            self.transformer_grid = tuple(grid_size)
            n_patches = 1
            for grid in grid_size:
                n_patches *= grid
            patch_size = 1
            self.hybrid = True
        else:
            raise NotImplementedError
            # patch_size = _triple(config.patches["size"])
            # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.hybrid_model_name = config.get('hybrid_model_name', 'ResNetV2')
        if self.hybrid:
            if self.hybrid_model_name == 'ResNetV2':
                self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,
                                             in_channels=config.in_channels)
                in_channels = self.hybrid_model.out_channel
            elif self.hybrid_model_name == 'ResNetV3':
                self.hybrid_model = ResNetV3(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,
                                             in_channels=config.in_channels, channel_factor=config.resnet.channel_factor)
                in_channels = self.hybrid_model.out_channel
            else:
                try:
                    self.hybrid_model = globals()[config.backbone.encoder](config.backbone)
                    in_channels = config.backbone.f_maps[-1]
                except Exception as e:
                    logger.error(e.args)
                    raise ModuleNotFoundError(self.hybrid_model_name)

        self.patch_embeddings = Conv3dReLU(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=2,
                                           stride=1)

        self.dropout = Dropout(config.transformer["dropout_rate"])

        self.learnable_pose_embeddings = False # self.config.get('learnable_pose_embeddings', True)

    def forward(self, xs):
        x, rois = xs
        x = x[0]
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # (1, 1024, 14, 14), patch size
        # (1, 768, 14, 14) , patch embedding = conv3d. 1024 ->768, vit 전에 linear-projection
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # 보간 ()

        if self.learnable_pose_embeddings:
            # (1, 768, 196) = (1, 768, 14*14)
            x = x.flatten(2)


            # (1, 196, 768)
            x = x.transpose(-1, -2)  # (B, n_patches, hidden)

            # (1, 196, 768) + (1, 196, 768)
            embeddings = x + self.position_embeddings
        else:
            # x = x.flatten(2)
            x = x.flatten(2).permute(2, 0, 1).contiguous()
            embeddings = x + position_embedding_sing3d_rois(rois, self.config.hidden_size // rois.shape[-1])
            # x = x.flatten(2)
            # embeddings = x.transpose(-1, -2)

        embeddings = self.dropout(embeddings)
        # features ( ([1, 512, 28, 28]), ([1, 256, 56, 56]), [1, 64, 112, 112]
        return embeddings, features



class TransformerInterpolatorV3(nn.Module):
    def __init__(self, config, img_size, vis):
        super(TransformerInterpolatorV3, self).__init__()
        self.embeddings01 = EmbeddingsInterpolator(config.query_model, img_size=img_size, in_channels=config.in_channels)

        self.stat_embedding_moduels = config.get('stat_embedding_moduels')

        # self.embeddings02 = EmbeddingsInterpolator(config.key_value_model, img_size=img_size, in_channels=config.in_channels)
        self.embeddings02 = EmbeddingsROIInterpolator(config.key_value_model)
        # if self.stat_embedding_moduels == 'EmbeddingsInterpolator':
        # else:
        #     self.embeddings02 = EmbeddingsInterpolator(config, img_size=img_size, in_channels=config.in_channels)
        self.encoder = EncoderQKV(config.query_model, vis)

    def forward(self, x1, x2):
        # x1, x2 = torch.split(input_ids, split_size_or_sections=2, dim=1)
        # embedding_output, features = self.embeddings(input_ids)
        embedding_q, features_q = self.embeddings01(x1)
        qmbedding_kv, features_kv = self.embeddings02(x2)
        # embedding_output = torch.cat([embedding_output01, embedding_output02], dim=1)
        ##
        encoded, attn_weights = self.encoder(embedding_q, qmbedding_kv)  # (B, n_patch, hidden)
        ##
        return encoded, attn_weights, (features_q, features_kv)


class VisionTransformerInterpolatorStatModel(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        super(VisionTransformerInterpolatorStatModel, self).__init__()
        self.num_classes = config.n_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = TransformerInterpolatorV3(config, config.img_size, vis)
        query_config = config.query_model
        self.decoder = DecoderInterpolatorV3(query_config)

        self.segmentation_head = SegmentationHead(
            in_channels=query_config['decoder_channels'][-1],
            out_channels=query_config['n_classes'],
            kernel_size=3,
        )
        self.final_activation = final_active_function[getattr(query_config, 'final_activation', 'tanh')]
        # torch.n
        self.config = config

    def forward(self, x1, x2):
        # x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        # torch.Size([1, 512, 28, 28]), torch.Size([1, 256, 56, 56]), torch.Size([1, 64, 112, 112])]
        x, attn_weights, (features_x1, features_x2) = self.transformer(x1, x2)  # (B, n_patch, hidden)
        # 보간법
        x = self.decoder(x, features_x1)
        logits = self.segmentation_head(x)
        logits = self.final_activation(logits)
        # logits = torch.tanh(logits)
        return logits


def position_embedding_sing3d(tensor: torch.Tensor,
                              num_pos_feats, normalize=False, scale=None, temperature=1000, eps=1e-6):
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

    assert tensor.ndim == 5
    assert not (scale is not None and normalize is False), "normalize should be True if scale is passed"
    if scale is None:
        scale = 2 * math.pi
    dtype = tensor.dtype
    device = tensor.device
    embeds = torch.meshgrid(*[torch.arange(i, dtype=dtype, device=device) for i in tensor.shape[2:]], indexing='ij')
    z_embed, y_embed, x_embed = [v[None] for v in embeds]
    # tensor axis is [z, y, x]

    if normalize:
        z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * scale
        y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * torch.div(dim_t,  2, rounding_mode='trunc') / num_pos_feats)

    pos_x = x_embed[:, :, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, :, None] / dim_t
    pos_z = z_embed[:, :, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
    pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
    pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
    pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
    return pos




class TeethNetEncoder(nn.Module):
    def __init__(self, confg):
        super(TeethNetEncoder, self).__init__()
        self._encoder_blocks = create_encoders(**confg)

    def forward(self, x):
        encoders_features = []
        for encoder in self._encoder_blocks:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        return x, encoders_features


class TeethNetDecoder(nn.Module):
    def __init__(self, config):
        super(TeethNetDecoder, self).__init__()

        hidden_size = config.hidden_size
        self.transformer_grid = config.patches.grid
        head_channels = config.f_maps[-1]
        self.conv_more = Conv3dReLU(
            hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

        self._decoder_blocks = create_decoders(**config)

    def forward(self, x, encoder_features):
        assert len(encoder_features) == len(self._decoder_blocks)

        B, n_patch, hidden = x.size()
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, *self.transformer_grid)
        target_shape = [v // 2 for v in encoder_features[0].size()[2:]]

        x = F.interpolate(x, target_shape, mode='trilinear', align_corners=True)

        x = self.conv_more(x)

        # encoders_features = []
        for enc_feat, decoder in zip(encoder_features, self._decoder_blocks):
            x = decoder(enc_feat, x)

        return x




def position_embedding_sing3d_rois(pose: torch.Tensor,
                              num_pos_feats, normalize=False, scale=1.0, temperature=1000, eps=1e-6):
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

    device = pose.device
    pose *= scale
    # embeds = torch.meshgrid(*[torch.arange(i, dtype=dtype, device=device) for i in tensor.shape[2:]])


    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * torch.div(dim_t,  2, rounding_mode='trunc') / num_pos_feats)
    pose_emb = pose[..., None] / dim_t
    # pose_emb = pose_emb[..., None] / dim_t
    pose_emb_stack = torch.stack([pose_emb[..., 0::2].sin(), pose_emb[..., 1::2].cos()], dim=-1)
    # dimmesnion 기준으로
    pose_emb_res = torch.cat(pose_emb_stack.split(1, dim=2), dim=-1).flatten(2)# .permute(0, 2, 1).contiguous()
    # pos = torch.cat((pos_z, pos_y, pos_x), dim=-1).permute(0, 2, 1).contiguous()
    return pose_emb_res