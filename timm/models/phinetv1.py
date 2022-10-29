import numpy as np
import torch
import torch.nn as nn

__all__ = ["PhiNet"]


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling

    Args:
        input_shape ([tuple/int]): [Input size]
        kernel_size ([tuple/int]): [Kernel size]

    Returns:
        [tuple]: [Padding coeffs]
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )


def preprocess_input(x, **kwargs):
    """Normalise channels between [-1, 1]

    Args:
        x ([Tensor]): [Contains the image, number of channels is arbitrary]

    Returns:
        [Tensor]: [Channel-wise normalised tensor]
    """

    return (x / 128.0) - 1


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute expansion factor based on the formula from the paper

    Args:
        t_zero ([int]): [initial expansion factor]
        beta ([int]): [shape factor]
        block_id ([int]): [id of the block]
        num_blocks ([int]): [number of blocks in the network]

    Returns:
        [float]: [computed expansion factor]
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks


class ReLUMax(torch.nn.Module):
    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.clamp(self.relu(x), max=self.max)


class HSwish(torch.nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()

    def forward(self, x):
        return x * nn.ReLU6(inplace=True)(x + 3) / 6


class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block"""

    def __init__(self, in_channels, out_channels, h_swish=True):
        """Constructor of SEBlock

        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            h_swish (bool, optional): [Whether to use the h_swish or not]. Defaults to True.
        """
        super(SEBlock, self).__init__()

        self.glob_pooling = lambda x: nn.functional.avg_pool2d(x, x.size()[2:])

        self.se_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        self.se_conv2 = nn.Conv2d(
            out_channels, in_channels, kernel_size=1, bias=False, padding="same"
        )

        if h_swish:
            self.activation = HSwish()
        else:
            self.activation = ReLUMax(6)

    def forward(self, x):
        """Executes SE Block

        Args:
            x ([Tensor]): [input tensor]

        Returns:
            [Tensor]: [output of squeeze-and-excitation block]
        """
        inp = x
        x = self.glob_pooling(x)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)
        x = x.expand_as(inp) * inp

        return x


class DepthwiseConv2d(torch.nn.Conv2d):
    """Depthwise 2D conv

    Args:
        torch ([Tensor]): [Input tensor for convolution]
    """

    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class SeparableConv2d(torch.nn.Module):
    """Implements SeparableConv2d"""

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.functional.relu,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        depth_multiplier=1,
    ):
        """Constructor of SeparableConv2d

        Args:
            in_channels ([int]): [Input number of channels]
            out_channels ([int]): [Output number of channels]
            kernel_size (int, optional): [Kernel size]. Defaults to 3.
            stride (int, optional): [Stride for conv]. Defaults to 1.
            padding (int, optional): [Padding for conv]. Defaults to 0.
            dilation (int, optional): []. Defaults to 1.
            bias (bool, optional): []. Defaults to True.
            padding_mode (str, optional): []. Defaults to 'zeros'.
            depth_multiplier (int, optional): [Depth multiplier]. Defaults to 1.
        """
        super().__init__()

        self._layers = torch.nn.ModuleList()

        depthwise = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding="valid",
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        bn = torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)

        self._layers.append(depthwise)
        self._layers.append(spatialConv)
        self._layers.append(bn)
        self._layers.append(activation)

    def forward(self, x):
        """Executes SeparableConv2d block

        Args:
            x ([Tensor]): [Input tensor]

        Returns:
            [Tensor]: [Output of convolution]
        """
        for l in self._layers:
            x = l(x)

        return x


class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block"""
    def __init__(self,
                 in_shape,
                 expansion,
                 stride,
                 filters,
                 has_se,
                 block_id=None,
                 res=True,
                 h_swish=True,
                 k_size=3,
                 dp_rate=0,
                 sd_p=1):
        """Defines the structure of the PhiNet conv block
        Args:
            in_shape ([Tuple]): [Input shape, as returned by Tensor.shape]
            expansion ([Int]): [Expansion coefficient]
            stride ([Int]): [Stride for conv block]
            filters ([Int]): [description]
            block_id ([Int]): [description]
            has_se (bool): [description]
            res (bool, optional): [description]. Defaults to True.
            h_swish (bool, optional): [description]. Defaults to True.
            k_size (int, optional): [description]. Defaults to 3.
            sd_p (float, optional): [stochastic depth probability]. Defaults to 0.
        """
        super(PhiNetConvBlock, self).__init__()
        self.skip_conn = False
        self.pdf = torch.distributions.bernoulli.Bernoulli(torch.Tensor([sd_p]))

        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]
        # Define activation function
        if h_swish:
            activation = HSwish()
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels, int(expansion * in_channels),
                kernel_size=1,
                padding="same",
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                int(expansion * in_channels),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)

        if stride == 2:
            pad = nn.ZeroPad2d(
                padding=correct_pad(in_shape, 3),
            )

            self._layers.append(pad)
        
        self._layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = int(expansion * in_channels) if block_id else in_channels
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding="same" if stride == 1 else "valid",
            # name=prefix + 'depthwise'
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = max(1, int(in_channels * 0.25))
            se_block = SEBlock(int(expansion * in_channels), num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=int(expansion * in_channels),
            out_channels=filters,
            kernel_size=1,
            padding="same",
            bias=False,
        )

        bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True

    def forward(self, x):
        """Executes PhiNet's convolutional block
        Args:
            x ([Tensor]): [Conv block input]
        Returns:
            [Tensor]: [Output of convolutional block]
        """
        if self.skip_conn:
            inp = x

        for l in self._layers:
            # print(l, l(x).shape)
            x = l(x)

        if self.skip_conn:
            if self.training:
                if torch.equal(self.pdf.sample(), torch.ones(1)):
                    return x + inp
                else:
                    return inp
            else:
                return x + inp  # can also be x*self.sd_p + inp
        else:
            return x


class PhiNet(nn.Module):
    def __init__(self, res=224, in_channels=3, B0=7, alpha=1.5, beta=1.0, t_zero=6, h_swish=False, squeeze_excite=True,
                 downsampling_layers=[5, 7], conv5_percent=0, first_conv_stride=2, first_conv_filters=48, b1_filters=24,
                 b2_filters=48, include_top=True, pooling=None, num_classes=1000, residuals=True, input_tensor=None, conv2d_input=False,
                 pool=False, drop_connect=0.2, drop_rate=0.2):
        """Generates PhiNets architecture
        Args:
            res (int, optional): [base network input resolution]. Defaults to 96.
            B0 (int, optional): [base network number of blocks]. Defaults to 7.
            alpha (float, optional): [base network width multiplier]. Defaults to 0.35.
            beta (float, optional): [shape factor]. Defaults to 1.0.
            t_zero (int, optional): [initial expansion factor]. Defaults to 6.
            h_swish (bool, optional): [Approximate Hswish activation - Enable for performance, disable for compatibility (gets replaced by relu6)]. Defaults to False.
            squeeze_excite (bool, optional): [SE blocks - Enable for performance, disable for compatibility]. Defaults to False.
            downsampling_layers (list, optional): [Indices of downsampling blocks (between 5 and B0)]. Defaults to [5,7].
            conv5_percent (int, optional): [description]. Defaults to 0.
            first_conv_stride (int, optional): [Downsampling at the network input - first conv stride]. Defaults to 2.
            first_conv_filters (int, optional): [description]. Defaults to 48.
            b1_filters (int, optional): [description]. Defaults to 24.
            b2_filters (int, optional): [description]. Defaults to 48.
            include_top (bool, optional): [description]. Defaults to True.
            pooling ([type], optional): [description]. Defaults to None.
            classes (int, optional): [description]. Defaults to 10.
            residuals (bool, optional): [disable residual connections to lower ram usage - residuals]. Defaults to True.
            input_tensor ([type], optional): [description]. Defaults to None.
        """
        super(PhiNet, self).__init__()
        p_l = drop_connect
        self.num_classes = num_classes
        self.classify = include_top
        num_blocks = round(B0)
        input_shape = (round(res), round(res), in_channels)
        if p_l != 1:
            prob_step = (1-p_l)/(round(B0) - 1)
            sd_p = np.arange(p_l, 1, prob_step)
            sd_p = sd_p[::-1]
        else:
            sd_p = [1]*(round(B0) - 1)
                
        self._layers = torch.nn.ModuleList()

        # Define self.activation function
        if h_swish:
            activation = HSwish()
        else:
            activation = ReLUMax(6)
            
        mp = nn.MaxPool2d((2, 2))
        
        if not conv2d_input:
            pad = nn.ZeroPad2d(
                padding=correct_pad(input_shape, 3),
            )

            self._layers.append(pad)

            sep1 = SeparableConv2d(
                in_channels,
                int(first_conv_filters * alpha),
                kernel_size=3,
                stride=(first_conv_stride, first_conv_stride),
                padding="valid",
                bias=False,
                activation=activation
            )

            self._layers.append(sep1)
            # self._layers.append(activation)

            block1 = PhiNetConvBlock(
                in_shape=(int(first_conv_filters * alpha), res / first_conv_stride, res / first_conv_stride),
                filters=int(b1_filters * alpha),
                stride=1,
                expansion=1,
                has_se=False,
                res=residuals,
                h_swish=h_swish,
                dp_rate=drop_rate
            )
            
            self._layers.append(block1)
        else:
            
            c1 = nn.Conv2d(
                in_channels,
                int(b1_filters*alpha),
                kernel_size=(3,3),
                bias=False
            )
            
            bn_c1 = nn.BatchNorm2d(int(b1_filters*alpha))
            
            self._layers.append(c1)
            self._layers.append(activation)
            self._layers.append(bn_c1)
        
        block2 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride, res / first_conv_stride),
            filters=int(b1_filters * alpha),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 1, num_blocks),
            block_id=1,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            sd_p=sd_p[0],
            dp_rate=drop_rate
        )
        
        block3 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride / 2, res / first_conv_stride / 2),
            filters=int(b1_filters * alpha),
            stride=1,
            expansion=get_xpansion_factor(t_zero, beta, 2, num_blocks),
            block_id=2,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            sd_p=sd_p[1],
            dp_rate=drop_rate
        )

        block4 = PhiNetConvBlock(
            (int(b1_filters * alpha), res / first_conv_stride / 2, res / first_conv_stride / 2),
            filters=int(b2_filters * alpha),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 3, num_blocks),
            block_id=3,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            sd_p=sd_p[2],
            dp_rate=drop_rate
        )

        self._layers.append(block2)
        if pool:
            self._layers.append(mp)
        self._layers.append(block3)
        self._layers.append(block4)
        if pool:
            self._layers.append(mp)

        
        block_id = 4
        block_filters = b2_filters
        spatial_res = res / first_conv_stride / 4
        in_channels_next = int(b2_filters * alpha)
        while num_blocks >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2
                if pool:
                    self._layers.append(mp)
            
            pn_block = PhiNetConvBlock(
                    (in_channels_next, spatial_res, spatial_res),
                    filters=int(block_filters * alpha),
                    stride=(2 if (block_id in downsampling_layers) and (not pool) else 1),
                    expansion=get_xpansion_factor(t_zero, beta, block_id, num_blocks),
                    block_id=block_id,
                    has_se=squeeze_excite,
                    res=residuals,
                    h_swish=h_swish,
                    k_size=(5 if (block_id / num_blocks) > (1 - conv5_percent) else 3),
                    sd_p=sd_p[block_id-2],
                    dp_rate=drop_rate
            )
            

            self._layers.append(pn_block)
            in_channels_next = int(block_filters * alpha)
            spatial_res = spatial_res / 2 if block_id in downsampling_layers else spatial_res
            block_id += 1

        if include_top:
            #Includes classification head if required
            self.glob_pooling = lambda x: nn.functional.avg_pool2d(x, x.size()[2:])
            self.class_conv2d = nn.Conv2d(
                int(block_filters * alpha),
                int(1280*alpha),
                kernel_size=1,
                bias=True
            )
            self.final_conv = nn.Conv2d(
                int(1280*alpha),
                num_classes,
                kernel_size=1,
                bias=True
            )


            # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # self.classifier = nn.Linear(int(block_filters * alpha), num_classes)
            # self.soft = nn.Softmax(dim=1)

    
    def forward(self, x):
        """Executes PhiNet network
        Args:
            x ([Tensor]): [input batch]
        """
        # i = 0
        for l in self._layers:
            # print("Layer ", i, l)
            x = l(x)
            # input(l)
            # input(x)
            # print("Output of layer ", i, x.shape)
            # i += 1

        if self.classify:
            x = self.glob_pooling(x)
            # input(x)
            x = self.final_conv(self.class_conv2d(x))
            # input(x)
            x = x.view(-1, x.shape[1])
            # input(x)
                        
            # return self.soft(x)
        
        return x
