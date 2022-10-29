import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class XiConv(nn.Module):
    # XiNET convolution
    def __init__(self, c1, c2, kernel_size=3, stride=1, padding=None, bias=False, groups=None, dilation=None, g=1, act=True, compression=4, attention=True, skip_tensor_in=None, skip_channels=1, pool=None, attention_k=3, attention_lite=True, batchnorm=True, dropout_rate=0,  skip_k=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.compression = compression
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c2//compression
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
 
        self.compression_conv = nn.Conv2d(c1, c2//compression, 1, 1,  groups=g, padding='same', bias=False)
        self.main_conv = nn.Conv2d(c2//compression if compression>1 else c1, c2, kernel_size, stride,  groups=g, padding='same' if stride==1 else autopad(kernel_size, padding), bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
 
        if attention:
            if attention_lite:
                self.att_pw_conv= nn.Conv2d(c2, self.attention_lite_ch_in, 1, 1, groups=g, padding='same', bias=False)
            self.att_conv = nn.Conv2d(c2 if not attention_lite else self.attention_lite_ch_in, c2, attention_k, 1, groups=g, padding='same', bias=False)
            self.att_act = nn.Sigmoid()
 
        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            self.skip_conv = nn.Conv2d(skip_channels, c2//compression, skip_k, 1,  groups=g, padding='same', bias=False)
        if batchnorm:
            self.bn = nn.BatchNorm2d(c2)
            self.bn0 = nn.BatchNorm2d(c2 // compression)
        if dropout_rate>0:
            self.do = nn.Dropout(dropout_rate)
 
 
 
    def forward(self, x):
        s = None
        # skip connection
        if isinstance(x, list):
            s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.skip_conv(s)
            x = x[0]
 
        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)
            # x = self.bn0(x)
            x = self.act(x)
 
        if s is not None:
            # print(f'Tensor shape {x.shape}')
            x = x+s
 
        if self.pool:
            x = self.mp(x)
        # main conv and activation
        x = self.main_conv(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)
 
        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in=self.att_pw_conv(x)
            else:
                att_in=x
            y = self.att_act(self.att_conv(att_in))
            x = x*y
 
 
        if self.dropout_rate > 0:
            x = self.do(x)
 
        # print(f'Output shape {x.shape} \n')
        return x

class EPConv(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=1,
        p=None,
        g=1,
        compression=4,
        pool=None,
        act=True,
        attention=True,
        attention_k=3,
        attention_lite=True,
        batchnorm=True,
        dropout_rate=0.,
        use_skip=None,
        skip_k=1,
        skip_channels=1,
        stoc_p=0.05,
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        # EPNET convolution
        super().__init__()
        # print(f"conf- c1={c1}, c2={c2}, k={k}, s={s}, g={g}")

        self.compression = compression
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = int(c2 // compression)
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.stoc_p = stoc_p

        self.compression_conv = nn.Conv2d(
            c1, int(c2 // compression), 1, 1, groups=g, padding="same", bias=False
        )
        self.bn_compress = nn.BatchNorm2d(int(c2 // compression))
        self.bn_compress1 = nn.BatchNorm2d(int(c2 // compression))
        self.main_conv = nn.Conv2d(
            int(c2 // compression) if compression > 1 else c1,
            c2,
            k,
            s,
            groups=g,
            padding="same" if s == 1 else (k - 1) // 2,
            bias=False,
        )
        self.act = (
            nn.ReLU(inplace=False)
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

        if attention:
            if attention_lite:
                self.att_pw_conv = nn.Conv2d(
                    c2,
                    self.attention_lite_ch_in,
                    1,
                    1,
                    groups=g,
                    padding="same",
                    bias=False,
                )
            self.att_conv = nn.Conv2d(
                c2 if not attention_lite else self.attention_lite_ch_in,
                c2,
                attention_k,
                1,
                groups=g,
                padding="same",
                bias=False,
            )
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)  # SNR after quantization

        if use_skip:
            self.skip_conv = nn.Conv2d(
                skip_channels,
                int(c2 // compression),
                skip_k,
                1,
                groups=g,
                padding="same",
                bias=False,
            )

        self.bn = nn.BatchNorm2d(c2)

        if dropout_rate > 0:
            self.do = nn.Dropout(dropout_rate, inplace=False)

    def forward(self, x):    
        s = None
        if isinstance(x, list):
            # print(x[0].shape[2:])
            s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.skip_conv(s)
            # s = self.bn_compress(s)
            # s = self.act(s)

            x = x[0]

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)
            # x = self.bn_compress1(x)
            # x = self.act(x)

        if not (s is None):
            if self.training:
                if self.stoc_p > 0:
                    with torch.no_grad():
                        r = torch.rand(1,)
                        r = (r < self.stoc_p).item()

                    if not r:
                        x = x + s
            else:
                x = x + s

        if self.pool:
            x = self.mp(x)

        # main conv and activation
        x = self.main_conv(x)
        x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in = self.att_pw_conv(x)
            else:
                att_in = x
            y = self.att_act(self.att_conv(att_in))
            x = x * y

        if self.dropout_rate > 0:
            x = self.do(x)          

        return x


class XNet(nn.Module):
    def __init__(
        self,
        a,
        g_init,
        b,
        N,
        downsample_layer,
        drop=0.2,
        pool=False,
        in_channels=3,
        num_classes=1000
    ):
        """Generates XNet architecture given hyperparms
        alpha * [8, 16, 32, 64, ...]
        layer_num [3, 7]
        compression_init = 6
        beta in [0.5, 1.2]
        compression = [1, 6, ..., beta * 6], beta < 1.0
        pool vs stride
        droput 0.1 / batchnorm

        :param a: _description_
        :type a: _type_
        :param c_init: _description_
        :type c_init: _type_
        :param b: _description_
        :type b: _type_
        :param N: _description_
        :type N: _type_
        :param drop: _description_
        :type drop: _type_
        :param pool: _description_
        :type pool: _type_
        """
        super().__init__()
        self.num_classes = num_classes
        N = int(N)
        channels_out = (np.array([2 ** (5 + i) for i in range((N))]) * a).astype(np.int32)
        channels_in = [in_channels] + list(channels_out)
        compression = [1, g_init]
        for _ in range(N - 2):
            step = (g_init * b - g_init) / (N - 2)
            compression.append(compression[-1] + step)

        self.cnn = nn.ModuleList(
            [
                EPConv(
                    channels_in[i],
                    channels_out[i],
                    compression=compression[i],
                    dropout_rate=.05,
                    pool=2 if (i in downsample_layer and pool) else None,
                    s=2 if (i in downsample_layer and not pool) else 1,
                    attention=False if i == 0 else True,
                    skip_channels=channels_out[0] if i != 0 else None,
                    use_skip=True if i != 0 else False,
                    k=5 if i == 0 else 3,
                )
                for i in range(N)
            ]
        )
        
        self.dnn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Linear(channels_out[-1], 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(),
            # nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            nn.Linear(channels_out[-1], self.num_classes),
        )

    def forward(self, x):
        skip = None

        for (idx, l) in enumerate(self.cnn):
            if not (skip is None):
                x = l([x, skip])
            else:
                x = l(x)
                skip = x.clone()

        return self.dnn(x)
