from torch import nn
from torch.nn import Sequential
from .common import conv, act, Concat, bn

# from .downsampler import Downsampler
import math

def get_module_names(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if name not in names:
            names.append(name)
    return names


def deepcopy_module(module, target):
    new_module = Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)                          # make new structure and,
            new_module[-1].load_state_dict(m.state_dict())         # copy weights
    return new_module


def soft_copy_param(target_link, source_link, tau):
    ''' soft-copy parameters of a link to another link. '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0-tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))




class ResBlock(nn.Module):
    def __init__(self, config, c_in, c_out, stride, dilation=(1,1), need1x1_up=False, residual=True):
        super().__init__()
        self.config = config
        self.stride = stride
        self.upsample = None
        self.residual = residual
        if stride == -2:
            self.upsample = nn.Upsample(scale_factor=-stride, mode=self.config.upsample_mode)
        if stride < 0:
            stride = 1

        self.downsample = None
        if stride > 1 or c_in != c_out:
            self.downsample = conv(
                c_in,
                c_out,
                config.filter_size_down,
                stride,
                bias=config.need_bias,
                pad=config.pad,
                downsample_mode=config.downsample_mode
            )

        self.conv1 = conv(
                c_in,
                c_out,
                config.filter_size_down,
                stride,
                bias=config.need_bias,
                pad=config.pad,
                downsample_mode=config.downsample_mode,
                dilation=dilation[0]
        )
        self.bn1 = bn(c_out)
        self.act1 = act(config.act_fun)
        self.conv2 = conv(
                c_out,
                c_out,
                config.filter_size_down,
                1,
                bias=config.need_bias,
                pad=config.pad,
                dilation=dilation[1])
        self.bn2 = bn(c_out)
        self.act2 = act(config.act_fun)

        if self.config.need1x1_up:
            self.conv3 = conv(
                c_out,
                c_out,
                1,
                bias=self.config.need_bias,
                pad=self.config.pad)
            self.bn3 = bn(c_out)
            self.act3 = act(config.act_fun)

    def forward(self, x):
        residual = x
        out = x
        if self.upsample:
            out = self.upsample(out)
            residual = self.upsample(residual)

        if self.downsample:
            residual = self.downsample(residual)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.config.need1x1_up:
            out = self.act2(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.residual:
                out += residual
            out = self.act3(out)
        else:
            if self.residual:
                out += residual
            out = self.act2(out)

        return out


class Config(object):
    input_channels = 32
    filter_size_down = 3
    bn_fun = "Batchbn"
    act_fun = "LeakyReLU"
    pad = "reflection"
    upsample_mode = "bilinear"
    downsample_mode = 'stride'
    need1x1_up = True
    output_channels = 3
    down_channels = [16, 32, 64, 64, 64, 128]
    down_stride = [1, 2, 2, 2, 2, 2]
    down_dilation = [1, 1, 1, 1, 1, 1]
    residual = True
    # down_dilation = [1, 2, 2, 2, 4, 4]

    need_bias = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DilatedResidualNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.config = Config(**kwargs)

        modules = [
            conv(
                self.config.input_channels,
                self.config.down_channels[0],
                kernel_size=1,
                stride=1,
                bias=self.config.need_bias,
                pad=self.config.pad),
            bn(self.config.down_channels[0]),
            act(self.config.act_fun),
        ]
        
        n_layers = len(self.config.down_channels)
        prev_channels = self.config.down_channels[0]
        
        for i, (channels, stride, dilation) in enumerate(zip(self.config.down_channels, self.config.down_stride, self.config.down_dilation)):
            residual = i >= 2  # No residual connection for degridding
            residual = self.config.residual and residual
            dilation = (dilation, dilation)
#             dilation = (dilation, 1 if (i % 2 == 0) or i <= n_layers - 2 else dilation)  # no dilation on first conv when on a new level
            modules.append(ResBlock(self.config, prev_channels, channels, stride, dilation, need1x1_up=self.config.need1x1_up, residual=residual))
            prev_channels = channels
            


        modules += [
            conv(
                prev_channels,
                self.config.output_channels,
                1,
                bias=self.config.need_bias,
                pad=self.config.pad
            ),
            nn.Sigmoid()
        ]

        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)
        return checkpoint_sequential(self.model, 3, x)
