# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class SegFormer2(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None):
        super(SegFormer2, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.backbone = backbone
        self.num_classes = num_classes
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.lc1_1 = layers.ConvBN(in_channels=c1_in_channels, out_channels=c1_in_channels, kernel_size=[5, 5],
                                   padding=2)
        self.lc1_2 = \
            layers.ConvBN(in_channels=c1_in_channels, out_channels=c1_in_channels, kernel_size=[5, 9], padding=[2, 4])
        self.lc1_3 = \
            layers.ConvBN(in_channels=c1_in_channels, out_channels=c1_in_channels, kernel_size=[9, 5], padding=[4, 2])
        self.lc1 = \
            layers.ConvBN(in_channels=3*c1_in_channels, out_channels=c1_in_channels, kernel_size=1, padding=0)
        
        self.lc2_1 = layers.ConvBN(in_channels=c2_in_channels, out_channels=c2_in_channels, kernel_size=[5, 5],
                                   padding=2)
        self.lc2_2 = \
            layers.ConvBN(in_channels=c2_in_channels, out_channels=c2_in_channels, kernel_size=[5, 9], padding=[2, 4])
        self.lc2_3 = \
            layers.ConvBN(in_channels=c2_in_channels, out_channels=c2_in_channels, kernel_size=[9, 5], padding=[4, 2])

        self.lc2 = \
            layers.ConvBN(in_channels=3*c2_in_channels, out_channels=c2_in_channels, kernel_size=1, padding=0)


        self.lc3_1 = layers.ConvBN(in_channels=c3_in_channels, out_channels=c3_in_channels, kernel_size=[5, 5],
                                   padding=2)
        self.lc3_2 = \
            layers.ConvBN(in_channels=c3_in_channels, out_channels=c3_in_channels, kernel_size=[5, 9], padding=[2, 4])
        self.lc3_3 = \
            layers.ConvBN(in_channels=c3_in_channels, out_channels=c3_in_channels, kernel_size=[9, 5], padding=[4, 2])
        self.lc3 = \
            layers.ConvBN(in_channels=3*c3_in_channels, out_channels=c3_in_channels, kernel_size=1, padding=0)
        
        self.lc4_1 = layers.ConvBN(in_channels=c4_in_channels, out_channels=c4_in_channels, kernel_size=[5, 5],
                                   padding=2)
        self.lc4_2 = \
            layers.ConvBN(in_channels=c4_in_channels, out_channels=c4_in_channels, kernel_size=[5, 9], padding=[2, 4])
        self.lc4_3 = \
            layers.ConvBN(in_channels=c4_in_channels, out_channels=c4_in_channels, kernel_size=[9, 5], padding=[4, 2])
        self.lc4 = \
            layers.ConvBN(in_channels=3*c4_in_channels, out_channels=c4_in_channels, kernel_size=1, padding=0)
        
        self.gelu = nn.GELU()

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            bias_attr=False)

        self.linear_pred = nn.Conv2D(
            embedding_dim, self.num_classes, kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        c1, c2, c3, c4 = feats

        ############## MLP decoder on C1-C4 ###########
        c1_shape = paddle.shape(c1)
        c2_shape = paddle.shape(c2)
        c3_shape = paddle.shape(c3)
        c4_shape = paddle.shape(c4)

        c1_1 = self.lc1_1(c1)
        c1_2 = self.lc1_2(c1)
        c1_3 = self.lc1_3(c1)
        c1 = self.dropout(self.gelu(self.lc1(paddle.concat([c1_1,c1_2,c1_3],axis=1))))
        
        #c1_1 = self.lc1_1_1(c1)
        #c1_2 = self.lc1_2_1(c1)
        #c1_3 = self.lc1_3_1(c1)
        #c1_4 = self.lc1_4_1(c1)
        #c1_5 = self.lc1_5_1(c1)
        #c1 = self.dropout(self.gelu(self.lc1_(paddle.concat([c1_1,c1_2,c1_3,c1_4,c1_5],axis=1))))

        c2_1 = self.lc2_1(c2)
        c2_2 = self.lc2_2(c2)
        c2_3 = self.lc2_3(c2)
        c2 = self.dropout(self.gelu(self.lc2(paddle.concat([c2_1,c2_2,c2_3],axis=1))))
        
        #c2_1 = self.lc2_1_1(c2)
        #c2_2 = self.lc2_2_1(c2)
        #c2_3 = self.lc2_3_1(c2)
        #c2_4 = self.lc2_4_1(c2)
        #c2_5 = self.lc2_5_1(c2)
        #c2 = self.dropout(self.gelu(self.lc2_(paddle.concat([c2_1,c2_2,c2_3,c2_4,c2_5],axis=1))))

        c3_1 = self.lc3_1(c3)
        c3_2 = self.lc3_2(c3)
        c3_3 = self.lc3_3(c3)        
        c3 = self.dropout(self.gelu(self.lc3(paddle.concat([c3_1,c3_2,c3_3],axis=1))))
        
       # c3_1 = self.lc3_1_1(c3)
        #c3_2 = self.lc3_2_1(c3)
       # c3_3 = self.lc3_3_1(c3)        
       # c3_4 = self.lc3_4_1(c3)
       # c3_5 = self.lc3_5_1(c3)
       # c3 = self.dropout(self.gelu(self.lc3_(paddle.concat([c3_1,c3_2,c3_3,c3_4,c3_5],axis=1))))

        c4_1 = self.lc4_1(c4)
        c4_2 = self.lc4_2(c4)
        c4_3 = self.lc4_3(c4)
        c4 = self.dropout(self.gelu(self.lc4(paddle.concat([c4_1,c4_2,c4_3],axis=1))))
        
       # c4_1 = self.lc4_1_1(c4)
       # c4_2 = self.lc4_2_1(c4)
       # c4_3 = self.lc4_3_1(c4)
       # c4_4 = self.lc4_4_1(c4)
       # c4_5 = self.lc4_5_1(c4)
      #  c4 = self.dropout(self.gelu(self.lc4_(paddle.concat([c4_1,c4_2,c4_3,c4_4,c4_5],axis=1))))
        
        ##c1_1_1 = self.lc1_1_1(c1)
        ##c1_2_1 = self.lc1_2_1(c1)
        ##c1_3_1 = self.lc1_3_1(c1)
        ##c1 = self.dropout(self.gelu(c1_1_1 + c1_2_1 + c1_3_1))

        ##c2_1_1 = self.lc2_1_1(c2)
        ##c2_2_1 = self.lc2_2_1(c2)
        ##c2_3_1 = self.lc2_3_1(c2)
        ##c2 = self.dropout(self.gelu(c2_1_1 + c2_2_1 + c2_3_1))

        ##c3_1_1 = self.lc3_1_1(c3)
        ##c3_2_1 = self.lc3_2_1(c3)
        ##c3_3_1 = self.lc3_3_1(c3)
        ##c3 = self.dropout(self.gelu(c3_1_1 + c3_2_1 + c3_3_1))

        ##c4_1_1 = self.lc4_1_1(c4)
        ##c4_2_1 = self.lc4_2_1(c4)
        ##c4_3_1 = self.lc4_3_1(c4)
        ##c4 = self.dropout(self.gelu(c4_1_1 + c4_2_1 + c4_3_1))

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        _c4 = F.interpolate(
            _c4,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(
            _c3,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(
            _c2,
            size=c1_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(
                logit,
                size=paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
