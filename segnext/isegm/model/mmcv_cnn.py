import warnings
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Replace MMCV registry with a simple identity decorator
def register_module():
    def _register_module(cls):
        return cls
    return _register_module

# Simple initialization functions to replace mmengine ones
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module, a=0, nonlinearity='relu', mode='fan_out', bias=0):
    nn.init.kaiming_normal_(
        module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

# Simple build functions
def build_conv_layer(cfg, *args, **kwargs):
    """Simple conv builder that returns nn.Conv2d by default."""
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    
    conv_type = cfg.get('type', 'Conv2d')
    if conv_type == 'Conv1d':
        return nn.Conv1d(*args, **kwargs)
    elif conv_type == 'Conv2d':
        return nn.Conv2d(*args, **kwargs)
    elif conv_type == 'Conv3d':
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported conv type: {conv_type}')

def build_norm_layer(cfg, num_features):
    """Simple norm layer builder."""
    if cfg is None:
        return 'bn', nn.BatchNorm2d(num_features)
    
    norm_type = cfg.get('type', 'BN')
    if norm_type == 'BN':
        norm_layer = nn.BatchNorm2d(num_features)
    elif norm_type == 'SyncBN':
        norm_layer = nn.SyncBatchNorm(num_features)
    elif norm_type == 'GN':
        num_groups = cfg.get('num_groups', 32)
        norm_layer = nn.GroupNorm(num_groups, num_features)
    elif norm_type == 'LN':
        norm_layer = nn.LayerNorm(num_features)
    elif norm_type == 'IN':
        norm_layer = nn.InstanceNorm2d(num_features)
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')
    
    return 'bn', norm_layer

def build_activation_layer(cfg):
    """Simple activation layer builder."""
    if cfg is None:
        return nn.ReLU(inplace=True)
    
    act_type = cfg.get('type', 'ReLU')
    inplace = cfg.get('inplace', True)
    
    if act_type == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(cfg.get('negative_slope', 0.01), inplace=inplace)
    elif act_type == 'PReLU':
        return nn.PReLU(cfg.get('num_parameters', 1))
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'Tanh':
        return nn.Tanh()
    elif act_type == 'Swish' or act_type == 'SiLU':
        return nn.SiLU(inplace=inplace)
    else:
        raise ValueError(f'Unsupported activation type: {act_type}')

def build_padding_layer(cfg, padding):
    """Simple padding layer builder."""
    if padding == 0:
        return nn.Identity()
    
    pad_type = cfg.get('type', 'zeros')
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    else:
        raise ValueError(f'Unsupported padding type: {pad_type}')


def efficient_conv_bn_eval_forward(bn: nn.modules.batchnorm._BatchNorm,
                                  conv: nn.modules.conv._ConvNd,
                                  x: torch.Tensor):
    """
    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.

    Args:
        bn (nn.modules.batchnorm._BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var +
                              bn.eps).reshape([-1] + [1] *
                                              (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() *\
        (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


@register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input.
        dilation (int | tuple[int]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to
            output channels.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): Padding mode.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers.
            Default: ('conv', 'norm', 'act').
        efficient_conv_bn_eval (bool): Whether use efficient conv when the
            consecutive bn is in eval mode. Default: `False`.
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act'),
                 efficient_conv_bn_eval: bool = False):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed if hasattr(self.conv, 'transposed') else False
        self.output_padding = self.conv.output_padding if hasattr(self.conv, 'output_padding') else 0
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(
                norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                                     nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # For customized conv layers without their own initialization
        # manners and PyTorch's conv layers, they will be initialized by
        # this method with default kaiming_init.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                # if the next operation is norm and we have a norm layer in
                # eval mode and we have enabled `efficient_conv_bn_eval` for
                # the conv operator, then activate the optimized forward and
                # skip the next norm operator since it has been fused
                if layer_index + 1 < len(self.order) and \
                        self.order[layer_index + 1] == 'norm' and norm and \
                        self.with_norm and not self.norm.training and \
                        self.efficient_conv_bn_eval_forward is not None:
                    self.conv.forward = partial(
                        self.efficient_conv_bn_eval_forward, self.norm,
                        self.conv)
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward
                else:
                    x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x

    def turn_on_efficient_conv_bn_eval(self, efficient_conv_bn_eval=True):
        # efficient_conv_bn_eval works for conv + bn
        # with `track_running_stats` option
        if efficient_conv_bn_eval and self.norm \
                            and isinstance(self.norm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) \
                            and self.norm.track_running_stats:
            self.efficient_conv_bn_eval_forward = efficient_conv_bn_eval_forward
        else:
            self.efficient_conv_bn_eval_forward = None

    @staticmethod
    def create_from_conv_bn(conv: nn.modules.conv._ConvNd,
                            bn: nn.modules.batchnorm._BatchNorm,
                            efficient_conv_bn_eval=True) -> 'ConvModule':
        """Create a ConvModule from a conv and a bn module."""
        self = ConvModule.__new__(ConvModule)
        super(ConvModule, self).__init__()

        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.inplace = False
        self.with_spectral_norm = False
        self.with_explicit_padding = False
        self.order = ('conv', 'norm', 'act')

        self.with_norm = True
        self.with_activation = False
        self.with_bias = conv.bias is not None

        # build convolution layer
        self.conv = conv
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed if hasattr(self.conv, 'transposed') else False
        self.output_padding = self.conv.output_padding if hasattr(self.conv, 'output_padding') else 0
        self.groups = self.conv.groups

        # build normalization layers
        self.norm_name, norm = 'bn', bn
        self.add_module(self.norm_name, norm)

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        return self