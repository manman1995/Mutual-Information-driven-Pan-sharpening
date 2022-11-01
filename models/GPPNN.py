from math import exp
import torch
from models.utils.CDC import cdcconv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules import InvertibleConv1x1
from models.refine import Refine,CALayer
import torch.nn.init as init



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)



def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class DenseBlockMscale(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier'):
        super(DenseBlockMscale, self).__init__()
        self.ops = DenseBlock(channel_in, channel_out, init)
        self.fusepool = nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc1 = nn.Sequential(nn.Conv2d(channel_out,channel_out,1,1,0),nn.LeakyReLU(0.1,inplace=True))
        self.fc2 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fc3 = nn.Sequential(nn.Conv2d(channel_out, channel_out, 1, 1, 0), nn.LeakyReLU(0.1, inplace=True))
        self.fuse = nn.Conv2d(3*channel_out,channel_out,1,1,0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')
        x1 = self.ops(x1)
        x2 = self.ops(x2)
        x3 = self.ops(x3)
        x2 = F.interpolate(x2, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        xattw = self.fusepool(x1+x2+x3)
        xattw1 = self.fc1(xattw)
        xattw2 = self.fc2(xattw)
        xattw3 = self.fc3(xattw)
        # x = x1*xattw1+x2*xattw2+x3*xattw3
        x = self.fuse(torch.cat([x1*xattw1,x2*xattw2,x3*xattw3],1))

        return x



def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlockMscale(channel_in, channel_out, init)
            else:
                return DenseBlockMscale(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)


        return out



class FeatureInteract(nn.Module):
    def __init__(self, channel_in, channel_split_num, subnet_constructor=subnet('DBNet'), block_num=4):
        super(FeatureInteract, self).__init__()
        operations = []

        # current_channel = channel_in
        channel_num = channel_in

        for j in range(block_num):
            b = InvBlock(subnet_constructor, channel_num, channel_split_num)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.fuse = nn.Conv2d((block_num-1)*channel_in,channel_in,1,1,0)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x, rev=False):
        out = x  # x: [N,3,H,W]
        outfuse = out
        for i,op in enumerate(self.operations):
            out = op.forward(out, rev)
            if i == 1:
                outfuse = out
            elif i > 1:
                outfuse = torch.cat([outfuse,out],1)
            # if i < 4:
            #     out = out+x
        outfuse = self.fuse(outfuse)

        return outfuse


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


class GPPNN(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat):
        super(GPPNN, self).__init__()
        self.extract_pan = FeatureExtract(pan_channels,n_feat//2)
        self.extract_ms = FeatureExtract(ms_channels,n_feat//2)

        # self.mulfuse_pan = Multual_fuse(n_feat//2,n_feat//2)
        # self.mulfuse_ms = Multual_fuse(n_feat // 2, n_feat // 2)

        self.interact = FeatureInteract(n_feat, n_feat//2)
        self.refine = Refine(n_feat, ms_channels)

    def forward(self, ms, i, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        mHR = upsample(ms, M, N)

        panf = self.extract_pan(pan)
        mHRf = self.extract_ms(mHR)

        feature_save(panf, '/home/jieh/Projects/PAN_Sharp/PansharpingMul/GPPNN/training/logs/GPPNN2/panf', i)
        feature_save(mHRf, '/home/jieh/Projects/PAN_Sharp/PansharpingMul/GPPNN/training/logs/GPPNN2/mHRf', i)

        finput = torch.cat([panf, mHRf], dim=1)
        fmid = self.interact(finput)
        HR = self.refine(fmid)+mHR

        return HR, panf, mHRf



import os
import cv2

def feature_save(tensor,name,i):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    tensor = torch.mean(tensor,dim=1)
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    inp = inp.squeeze(2)
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    if not os.path.exists(name):
        os.makedirs(name)
    # for i in range(tensor.shape[1]):
    #     inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
    #     inp = np.clip(inp,0,1)
    # # inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
    #
    #     cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(name + '/' + str(i) + '.png', inp)


class EdgeBlock(nn.Module):
    def __init__(self, channelin, channelout):
        super(EdgeBlock, self).__init__()
        self.process = nn.Conv2d(channelin,channelout,3,1,1)
        self.Res = nn.Sequential(nn.Conv2d(channelout,channelout,3,1,1),
            nn.ReLU(),nn.Conv2d(channelout, channelout, 3, 1, 1))
        self.CDC = cdcconv(channelin, channelout)

    def forward(self,x):

        x = self.process(x)
        out = self.Res(x) + self.CDC(x)

        return out

class FeatureExtract(nn.Module):
    def __init__(self, channelin, channelout):
        super(FeatureExtract, self).__init__()
        self.conv = nn.Conv2d(channelin,channelout,1,1,0)
        self.block1 = EdgeBlock(channelout,channelout)
        self.block2 = EdgeBlock(channelout, channelout)

    def forward(self,x):
        xf = self.conv(x)
        xf1 = self.block1(xf)
        xf2 = self.block2(xf1)

        return xf2


from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
CE = torch.nn.BCELoss(reduction='sum')


class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size = 4):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        # self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc1_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_depth1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # 
        # self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc1_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_depth2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        self.fc1_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc1_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)
        self.fc2_depth3 = nn.Linear(channels * 1 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        depth_feat = self.layer4(self.leakyrelu(self.layer2(depth_feat)))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        # if rgb_feat.shape[2] == 16:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 16 * 16)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 16 * 16)
        #
        #     mu_rgb = self.fc1_rgb1(rgb_feat)
        #     logvar_rgb = self.fc2_rgb1(rgb_feat)
        #     mu_depth = self.fc1_depth1(depth_feat)
        #     logvar_depth = self.fc2_depth1(depth_feat)
        # elif rgb_feat.shape[2] == 22:
        #     rgb_feat = rgb_feat.view(-1, self.channel * 1 * 22 * 22)
        #     depth_feat = depth_feat.view(-1, self.channel * 1 * 22 * 22)
        #     mu_rgb = self.fc1_rgb2(rgb_feat)
        #     logvar_rgb = self.fc2_rgb2(rgb_feat)
        #     mu_depth = self.fc1_depth2(depth_feat)
        #     logvar_depth = self.fc2_depth2(depth_feat)
        # else:
        rgb_feat = rgb_feat.view(-1, self.channel * 1 * 32 * 32)
        depth_feat = depth_feat.view(-1, self.channel * 1 * 32 * 32)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_depth = self.fc1_depth3(depth_feat)
        logvar_depth = self.fc2_depth3(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)
        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth+ce_depth_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss




###########################################################################################################



# class Multual_fuse(nn.Module):
#     def __init__(self, in_channels, channels):
#         super(Multual_fuse, self).__init__()
#         self.convx = nn.Conv2d(in_channels,channels,3,1,1)
#         self.fuse = CALayer(channels*2,4)
#         self.convout = nn.Conv2d(channels*2,channels,3,1,1)
#
#     def tile(self, a, dim, n_title):
#         init_dim = a.size(dim)
#         repeat_idx = [1] * a.dim()
#         repeat_idx[dim] = n_title
#         a = a.repeat(*(repeat_idx))
#         order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_title) + i for i in range(init_dim)])).cuda()
#         return torch.index_select(a,dim,order_index)
#
#
#     def forward(self,x,y):
#         x = self.convx(x)
#         y = torch.unsqueeze(y, 2)
#         y = self.tile(y, 2, x.shape[2])
#         y = torch.unsqueeze(y, 3)
#         y = self.tile(y, 3, x.shape[3])
#         fusef = self.fuse(torch.cat([x,y],1))
#         out = self.convout(fusef)
#
#         return out

