# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import torch.nn.functional as F
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from functools import reduce

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        assert config.hidden_size % self.num_attention_heads == 0,  'hidden_size % num_heads == 0'
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

    def forward(self, hidden_states):
        # (1, 196, 768)
        # simliarity(q,k) = Q*k_t/ scaling
        # num_head = 12
        # sacling = sqrt(attention_head_size), attention_head_size 64 = hidden_size / num_head = 768 / 12
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

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


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _triple(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1],
                          img_size[2] // 16 // grid_size[2])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16, patch_size[2] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) *\
                        (img_size[1] // patch_size_real[1]) *\
                        (img_size[2] // patch_size_real[2])
            self.hybrid = True
        else:
            patch_size = _triple(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor,
                                         in_channels=in_channels)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # (1, 14*14, 768) = (1, 196, 768)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # (1, 1024, 14, 14), patch size
        # (1, 768, 14, 14) , patch embedding = conv3d. 1024 ->768, vit 전에 linear-projection
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # (1, 768, 196) = (1, 768, 14*14)
        x = x.flatten(2)
        # (1, 196, 768)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # (1, 196, 768) + (1, 196, 768)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        # features ( ([1, 512, 28, 28]), ([1, 256, 56, 56]), [1, 64, 112, 112]
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        # (1, 196, 768)
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)

        # #layer = 12 // assert len(self.layer) == 12

        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, in_channels=config.in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)



class InterpolateUpsampling(nn.Module):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, scale_factor=2, mode='trilinear'):
        super(InterpolateUpsampling, self).__init__()

        self.mode = mode
        self.scale_factor = scale_factor


    def forward(self, x):
        # output_size = np.array([*x.size()[2:]]) * self.scale_factor
        output_size = torch.tensor(x.shape[2:], device=x.device) * self.scale_factor
        return F.interpolate(x, size=tuple(output_size), mode=self.mode, align_corners=True)



class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.up = InterpolateUpsampling(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = InterpolateUpsampling(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # upsampling = F.
        super().__init__(conv3d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 128
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

        if self.config.n_skip != 0:
            pass
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # d = h = w = round(np.power(n_patch, 1/3))# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        d = h = w = torch.round(torch.pow(torch.tensor(n_patch), 1 / 3)).to(torch.int64)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # (1, 196, 768) -? (1, 14x14, 768) // transpose -> (1, 768, 14x14) . 14 = 224 / 16
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, d, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            # print('x', x.shape, '\t skip', skip.shape if skip else 'None', next(decoder_block.parameters()).shape)
            x = decoder_block(x, skip=skip)
            # print('decoder', x.shape)
        return x


class SoftMax:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input):
        return  torch.softmax(input, dim=self.dim)

final_active_function = {
    'tanh': torch.tanh,
    'softmax': SoftMax(dim=1),
    'sigmoid': torch.sigmoid
}

class VisionTransformer(nn.Module):
    def __init__(self, config, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = config.n_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, config.img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.final_activation = final_active_function[getattr(config, 'final_activation', 'tanh')]
        # torch.n
        self.config = config

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        # torch.Size([1, 512, 28, 28]), torch.Size([1, 256, 56, 56]), torch.Size([1, 64, 112, 112])]
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = self.final_activation(logits)
        # logits = torch.tanh(logits)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class VisionTransformerMask(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(VisionTransformerMask, self).__init__(*args, **kwargs)

    def forward(self, x, mask):
        # mask not used
        return super(VisionTransformerMask, self).forward(x)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


class EmbeddingInterpolator(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(EmbeddingInterpolator, self).__init__()
        self.hybrid = None
        self.config = config

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                     width_factor=config.resnet.width_factor,
                                     in_channels=in_channels)
        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=(1, 1, 1),
                                       stride=(1, 1, 1))
        self.grid_size = self.config.patches.grid
        def func(x,y):
            return x*y
        n_patches = reduce(func, self.config.patches.grid)

        # (1, 14*14, 768) = (1, 196, 768)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(x)
        # (1, 1024, 14, 14), patch size
        # (1, 768, 14, 14) , patch embedding = conv3d. 1024 ->768, vit 전에 linear-projection
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))

        # interpolation
        target_size = torch.tensor(*[self.grid_size])
        x = F.interpolate(x, tuple(target_size), mode='trilinear')

        # (1, 768, 196) = (1, 768, 14*14)
        x = x.flatten(2)
        # (1, 196, 768)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # (1, 196, 768) + (1, 196, 768)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        # features ( ([1, 512, 28, 28]), ([1, 256, 56, 56]), [1, 64, 112, 112]
        return embeddings, features


class DecoderCupInterpolator(DecoderCup):
    def __init__(self, config):
        super(DecoderCupInterpolator, self).__init__(config)
        self.grid_size = config.patches.grid

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        # d = h = w = round(np.power(n_patch, 1/3))# reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        d = h = w = ([self.grid_size]*3).to(torch.int64)
            # d = h = w = torch.round(torch.pow(torch.tensor(n_patch), 1 / 3)).to(torch.int64)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))

        # interpolation
        target_size = torch.tensor(*[self.config.patches.grid])
        x = F.interpolate(hidden_states, tuple(target_size), mode='trilinear')

        # (1, 196, 768) -? (1, 14x14, 768) // transpose -> (1, 768, 14x14) . 14 = 224 / 16
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, d, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            # print('x', x.shape, '\t skip', skip.shape if skip else 'None', next(decoder_block.parameters()).shape)
            x = decoder_block(x, skip=skip)
            # print('decoder', x.shape)
        return x


class TransformerInterpolator(Transformer):
    def __init__(self, config, img_size, vis):
        super(TransformerInterpolator, self).__init__(config, img_size, vis)
        self.embeddings = EmbeddingInterpolator(config, img_size=img_size, in_channels=config.in_channels)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class VisionTransformerInterpolator(VisionTransformer):
    def __init__(self, config, zero_head=False, vis=False):
        super(VisionTransformerInterpolator, self).__init__(config, zero_head, vis)
        self.transformer = TransformerInterpolator(config, config.img_size, vis)
        self.decoder = DecoderCupInterpolator(config)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        # torch.Size([1, 512, 28, 28]), torch.Size([1, 256, 56, 56]), torch.Size([1, 64, 112, 112])]
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        logits = self.final_activation(logits)
        # logits = torch.tanh(logits)
        return logits
