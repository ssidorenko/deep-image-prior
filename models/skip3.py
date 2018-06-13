import torch
import torch.nn as nn
from .common import *

def skip3(
        num_input_channels=2, num_output_channels=3, 
        num_channels_up=[128, 128, 64, 32, 16], filter_size_up=3,
        num_channels_skip=[4, 4, 4, 4, 4], filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_up)

    n_scales = len(num_channels_up) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = nn.Sequential()
    input_depth = num_input_channels
    for i in range(n_scales)[::-1]:
#         print("iteration {} start".format(i))
#         print(model_tmp)
        inner = nn.Sequential()
        skip = nn.Sequential()
        model_tmp = nn.Sequential(inner, model_tmp)

        k = (num_channels_up[i - 1]) if i > 0 else input_depth
        
        if num_channels_skip[i] != 0:
            model_tmp = Concat(1, skip, model_tmp)
            skip.add(nn.Upsample(scale_factor=2**(n_scales-i), mode=upsample_mode[i]))
            skip.add(conv(k, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
#         print("iteration {} after skip".format(i))
#         print(model_tmp)
        inner.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        inner.add(conv(k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        inner.add(bn(num_channels_up[i]))
        inner.add(act(act_fun))
        
#         inner.add(conv(num_channels_up[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
#         inner.add(bn(num_channels_up[i]))
#         inner.add(act(act_fun))

#         inner.add(conv(num_channels_up[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
#         inner.add(bn(num_channels_up[i]))
#         inner.add(act(act_fun))
        
        if need1x1_up:
            inner.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            inner.add(bn(num_channels_up[i]))
            inner.add(act(act_fun))

    model.add(model_tmp)
    model.add(conv(num_channels_up[-1] + sum(num_channels_skip), num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
