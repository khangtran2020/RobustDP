import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn.utils import spectral_norm
from Models.modules.spectral_norm import spectral_norm_conv
from Utils.console import console

class CNN(nn.Module):

    def __init__(self, channel:List, hid_dim:List, img_size:int, channel_in:int, out_dim:int, debug:int=1, kernal_size:int=5, 
                 padding:int=0, stride:int=1, dropout:float=0.2):
        super(CNN, self).__init__()

        # general
        self.layer_name = []
        self.lay_out_size = {}
        self.inter_act = nn.ReLU()
        if debug:
            self.img_slist = []

        # conv part
        img_s = img_size
        if debug:
            self.img_slist.append(img_s)
        self.cnn_layers = nn.ModuleList()

        self.cnn_layers.append(
            spectral_norm_conv(nn.Conv2d(channel_in, channel[0], kernel_size=kernal_size, padding=padding, stride=stride), debug=bool(debug))
        )
        img_s = int((img_s + 2*padding - 1*(kernal_size - 1) - 1) / stride + 1)
        self.num_trans = 1
        if debug:
            self.img_slist.append(img_s)
        self.layer_name.append('conv_1')
        self.lay_out_size['conv_1'] = img_s*img_s*channel[0]

        self.cnn_layers.append(nn.MaxPool2d(2))
        self.layer_name.append(f'pool_1')
        img_s = int((img_s + 2*padding - 1*(2 - 1) - 1) / 2 + 1)
        if debug:
            self.img_slist.append(img_s)

        for i in range(1, len(channel)):
            self.cnn_layers.append(
                spectral_norm_conv(nn.Conv2d(channel[i-1], channel[i], kernel_size=kernal_size, padding=padding, stride=stride), debug=bool(debug))
            )
            self.num_trans += 1
            self.layer_name.append(f'conv_{i+1}')
            img_s = int((img_s + 2*padding - 1*(kernal_size - 1) - 1) / stride + 1)
            self.lay_out_size[f'conv_{i+1}'] = img_s*img_s*channel[0]
            if debug:
                self.img_slist.append(img_s)
            self.cnn_layers.append(nn.MaxPool2d(2))
            self.layer_name.append(f'pool_{i+1}')
            img_s = int((img_s + 2*padding - 1*(2 - 1) - 1) / 2 + 1)
            if debug:
                self.img_slist.append(img_s)

        self.flatten_size = img_s*img_s*channel[-1]

        # linear part
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(spectral_norm(nn.Linear(self.flatten_size, hid_dim[0])))
        self.layer_name.append('linr_1')
        self.num_trans += 1
        for i in range(1, len(hid_dim)):
            self.linear_layers.append(spectral_norm(nn.Linear(hid_dim[i-1], hid_dim[i])))
            self.num_trans += 1
            self.layer_name.append(f'linr_{i+1}')
        self.last_lay = nn.Linear(hid_dim[-1], out_dim, bias=False)
        self.conv_drop = nn.Dropout2d(dropout)
        self.linr_drop = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.cnn_layers):
            name = self.layer_name[i]
            if ('conv' in name) & (i > 0):
                self.conv_drop(h)
            h = layer(h)
            if ('pool' in name) or (i==0):
                h = self.inter_act(h)

        h = h.view(-1, self.flatten_size)
        for i, layer in enumerate(self.linear_layers):
            self.linr_drop(h)
            h = layer(h)
        h = self.linr_drop(h)
        h = self.last_lay(h)
        return h
    