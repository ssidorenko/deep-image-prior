import torch
import torch.nn as nn
from .common import *

def flat_net(
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
    input_depth = num_input_channels
    for i in range(len(num_channels_up)):

#         model.add(bn(k))
        model.add(conv(input_depth, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model.add(bn(num_channels_up[i]))
        model.add(act(act_fun))

        model.add(conv(num_channels_up[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model.add(bn(num_channels_up[i]))
        model.add(act(act_fun))

        
        if need1x1_up:
            model.add(conv(num_channels_up[i], num_channels_up[i], 1, 1, bias=need_bias, pad=pad))
            model.add(bn(num_channels_up[i]))
            model.add(act(act_fun))
        input_depth = num_channels_up[i]

    model.add(conv(num_channels_up[-1], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
