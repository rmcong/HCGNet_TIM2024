import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
from base_network import *
from torchvision.transforms import *


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        ####
        elif scale_factor == 16:
            kernel = 20
            stride = 16
            padding = 2

        self.scale_factor = scale_factor

        # Initial Feature Extraction
        self.feat0 = FeatureExtract1(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat2 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat3 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat4 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.feat_color0 = FeatureExtract1(3, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color1 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color2 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color3 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color4 = FeatureExtract2(base_filter, 3, 1, 1, activation='prelu', norm=None)

        # MCE
        self.m1 = MultiBlock5(5 * 64, kernel, stride, padding)
        self.m2 = MultiBlock4(4 * 64, kernel, stride, padding)
        self.m3 = MultiBlock3(3 * 64, kernel, stride, padding)
        self.m4 = MultiBlock2(2 * 64, kernel, stride, padding)
        self.m5 = MultiBlock1(64, kernel, stride, padding)

        # LDE
        self.f = LF_Mask3(64, base_filter, 3, 1, 1)
        self.f1 = LF_Mask3(64, base_filter, 3, 1, 1)

        # HAG
        self.h1 = HighFusion(2 * 64, base_filter, 3, 1, 1)
        self.h2 = HighFusion(2 * 64, base_filter, 3, 1, 1)
        self.h3 = HighFusion(2 * 64, base_filter, 3, 1, 1)
        self.h4 = HighFusion(2 * 64, base_filter, 3, 1, 1)
        self.h5 = HighFusion(2 * 64, base_filter, 3, 1, 1)

        self.s1 = SpatialAttention()
        self.s2 = SpatialAttention()
        self.s3 = SpatialAttention()
        self.s4 = SpatialAttention()
        self.s5 = SpatialAttention()

        # AFP 1
        self.r1_1 = AAP_alter1(scale_factor, 64, base_filter, kernel, stride, padding)
        self.r1_2 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r1_3 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r1_4 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        # self.r1_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        # AFP 2
        self.r2_1 = AAP_alter1(scale_factor, 64, base_filter, kernel, stride, padding)
        self.r2_2 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r2_3 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r2_4 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        # self.r2_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        # AFP 3
        self.r3_1 = AAP_alter1(scale_factor, 64, base_filter, kernel, stride, padding)
        self.r3_2 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r3_3 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r3_4 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        # self.r3_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        # AFP 4
        self.r4_1 = AAP_alter1(scale_factor, 64, base_filter, kernel, stride, padding)
        self.r4_2 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r4_3 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r4_4 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        # self.r4_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        # AFP 5
        self.r5_1 = AAP_alter1(scale_factor, 64, base_filter, kernel, stride, padding)
        self.r5_2 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r5_3 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        self.r5_4 = AAP_alter2(scale_factor, base_filter, base_filter, kernel, stride, padding)
        # self.r5_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        self.down1 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation=None, norm=None)
        self.down2 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation=None, norm=None)
        self.down3 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation=None, norm=None)
        self.down4 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation=None, norm=None)
        self.down5 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation=None, norm=None)
        self.down_LF1 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down_LF2 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.up1 = DeconvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.up = DeconvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        # Reconstruction

        self.output_conv1_1 = ConvBlock(4 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv2_1 = ConvBlock(4 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv3_1 = ConvBlock(4 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv4_1 = ConvBlock(4 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv5_1 = ConvBlock(4 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.guide_5 = ConvBlock(2 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        # self.guide_4 = ConvBlock(2 * base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_pixshuff = Upsampler(scale_factor, base_filter)
        self.output_conv = ConvBlock(base_filter, num_channels, 1, 1, 0, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb, depth):

        x = self.feat0(depth)
        x1 = self.feat1(x)
        x2 = self.feat2(x1)
        x3 = self.feat3(x2)
        x4 = self.feat4(x3)

        c = self.feat_color0(rgb)
        c1 = self.feat_color1(c)

        c2 = self.feat_color2(c1)
        c3 = self.feat_color3(c2)
        c4 = self.feat_color4(c3)

        ############5
        mv5 = self.m5(x4)
        rb1 = self.r5_1(mv5)
        rb2 = self.r5_2(rb1)
        rb3 = self.r5_3(rb2)
        rb4 = self.r5_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4), 1)
        r5 = self.output_conv5_1(concat_h)
        s_att5 = self.s5(c4)
        hs5 = c4 + c4 * s_att5


        g5 = torch.cat((hs5, r5), 1)
        temp5 = self.h5(g5)
        d5 = r5 + temp5 * r5

        x4_2 = self.down5(d5)
        d5 = x4_2 + x4

        ##############4
        x3_1 = torch.cat((x3, d5), 1)
        mv4 = self.m4(x3_1)
        rb1 = self.r4_1(mv4)
        rb2 = self.r4_2(rb1)
        rb3 = self.r4_3(rb2)
        rb4 = self.r4_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4), 1)
        r4 = self.output_conv4_1(concat_h)
        s_att4 = self.s4(c4)
        hs4 = c4 + c4 * s_att4

        g4 = torch.cat((hs4, r4), 1)
        temp4 = self.h4(g4)
        d4 = r4 + temp4 * r4

        x3_2 = self.down4(d4)
        d4 = x3_2 + x3

        ##############3
        x2_1 = torch.cat((x2, d4, d5), 1)
        mv3 = self.m3(x2_1)
        rb1 = self.r3_1(mv3)
        rb2 = self.r3_2(rb1)
        rb3 = self.r3_3(rb2)
        rb4 = self.r3_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4), 1)
        r3 = self.output_conv3_1(concat_h)
        s_att3 = self.s3(c4)
        hs3 = c4 + c4 * s_att3

        g3 = torch.cat((hs3, r3), 1)
        temp3 = self.h3(g3)
        d3 = r3 + temp3 * r3

        x2_2 = self.down3(d3)
        d3 = x2_2 + x2

        ##############2
        xx1 = self.up1(x1)
        f1 = self.f1(c1, xx1)
        x1_1 = self.down_LF2(f1)

        x1_1 = torch.cat((x1_1, d3, d4, d5), 1)
        mv2 = self.m2(x1_1)
        rb1 = self.r2_1(mv2)
        rb2 = self.r2_2(rb1)
        rb3 = self.r2_3(rb2)
        rb4 = self.r2_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4), 1)
        r2 = self.output_conv2_1(concat_h)
        s_att2 = self.s2(c4)
        hs2 = c4 + c4 * s_att2

        g2 = torch.cat((hs2, r2), 1)
        temp2 = self.h2(g2)
        d2 = r2 + temp2 * r2

        x1_2 = self.down2(d2)
        d2 = x1_2 + x1

        ##############1
        xx = self.up(x)
        f = self.f(c, xx)
        x_1 = self.down_LF1(f)

        x_1 = torch.cat((x_1, d2, d3, d4, d5), 1)
        mv1 = self.m1(x_1)
        rb1 = self.r1_1(mv1)
        rb2 = self.r1_2(rb1)
        rb3 = self.r1_3(rb2)
        rb4 = self.r1_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4), 1)
        r1 = self.output_conv1_1(concat_h)
        s_att1 = self.s1(c4)
        hs1 = c4 + c4 * s_att1

        g1 = torch.cat((hs1, r1), 1)
        temp1 = self.h1(g1)
        d1 = r1 + temp1 * r1

        x_2 = self.output_pixshuff(x)
        d1 = d1 + x_2
        d = self.output_conv(d1)
        d = d
        return d
