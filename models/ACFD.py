import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import torch.nn.functional as F
from zoedepth.models.base_models.midas import MidasCore
from torchvision.ops import DeformConv2d

class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.sim_bis1 = nn.Parameter(torch.ones(1))
        self.sim_bis2 = nn.Parameter(torch.ones(1))
        self.sim_bis3 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_sort, sim_sort_idx = sim.sort(dim=1, descending=True)

        mask_max = torch.zeros_like(sim)
        mask_1 = torch.zeros_like(sim)
        mask_2 = torch.zeros_like(sim)
        mask_3 = torch.zeros_like(sim)

        mask_max.scatter_(1, sim_sort_idx[:, 0:1, :], 1.)
        mask_1.scatter_(1, sim_sort_idx[:, 1:2, :], 1.)
        mask_2.scatter_(1, sim_sort_idx[:, 2:3, :], 1.)
        mask_3.scatter_(1, sim_sort_idx[:, 3:4, :], 1.)

        sim_max = sim * mask_max
        sim_1 = sim * mask_1
        sim_2 = sim * mask_2
        sim_3 = sim * mask_3
        sim = sim_max + self.sim_bis1*sim_1 + self.sim_bis2*sim_2 + self.sim_bis3*sim_3

        value2 = rearrange(value, 'b c w h -> b (w h) c')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    sim.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):

    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class ACFDnet(nn.Module):
    def __init__(self, max_depth=10.0):
        super().__init__()
        self.max_depth = max_depth
        self.decoder = ELMDecoder()

        channels_out = 64
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),  # 64 64
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))  # 64 1
        self.encoder = BEiTEncoder()

    def forward(self, x):

        rel_depth, output = self.encoder(x)

        out, out_4, out_8, out_16 = self.decoder(output[4], output[3], output[2], output[1])
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth
        out_4 = torch.sigmoid(out_4) * self.max_depth
        out_8 = torch.sigmoid(out_8) * self.max_depth
        out_16 = torch.sigmoid(out_16) * self.max_depth

        return {'pred_d': out_depth, 'pred_1/4': out_4, 'pred_1/8': out_8, 'pred_1/16': out_16}


class SoftTransformer(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24):
        super(SoftTransformer, self).__init__()
        self.dim = dim
        self.softtrans = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.fenpin = Fenpin(dim)

    def forward(self, input):

        cluste_tensor = input + self.drop_path(self.softtrans(self.norm1(input)))
        Enhance_out = self.fenpin(cluste_tensor)
        output_tensor = Enhance_out + self.drop_path(self.mlp(self.norm2(Enhance_out)))

        return output_tensor


class BEiTEncoder(nn.Module):
    def __init__(self, encoder_lr_factor=10, **kwargs):
        super().__init__()

        core = MidasCore.build(midas_model_type="DPT_BEiT_L_384", use_pretrained_midas=True,
                               train_midas=False, fetch_features=True, freeze_bn=True, **kwargs)
        self.core = core
        self.encoder_lr_factor = encoder_lr_factor

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(
                freeze_rel_pos=self.pos_enc_lr_factor <= 0)

    def forward(self, x, denorm=False):
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)
        return rel_depth, out



class ELMDecoder(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256, 512], depths=[3, 8, 27, 3], norm_layer=nn.LayerNorm,
                 layers=[6, 6, 24, 6],
                 mlp_ratios=[8, 8, 4, 4],
                 act_layer=nn.GELU, norm=nn.BatchNorm2d,
                 drop_rate=0., drop_path_rate=0.,
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[8, 8, 16, 16], head_dim=[32, 32, 32, 32]):
        super().__init__()
        self.ELM_256 = SoftTransformer(embed_dims[2], mlp_ratios[2],
                                       act_layer, norm,
                                       drop_rate, drop_path_rate,
                                       proposal_w[2], proposal_h[2], fold_w[2], fold_h[2], heads[2], head_dim[2])
        self.ELM_128 = SoftTransformer(embed_dims[1], mlp_ratios[1],
                                       act_layer, norm,
                                       drop_rate, drop_path_rate,
                                       proposal_w[1], proposal_h[1], fold_w[1], fold_h[1], heads[1], head_dim[1])
        self.ELM_64 = SoftTransformer(embed_dims[0], mlp_ratios[0],
                                      act_layer, norm,
                                      drop_rate, drop_path_rate,
                                      proposal_w[0], proposal_h[0], fold_w[0], fold_h[0], heads[0], head_dim[0])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv_256 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv_128 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv_64 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.compress_4 = nn.Sequential(
            nn.Conv2d(in_channels=448,
                      out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.compress_8 = nn.Sequential(
            nn.Conv2d(in_channels=448,
                      out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.compress_16 = nn.Sequential(
            nn.Conv2d(in_channels=448,
                      out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.last_layer_depth_16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

        self.last_layer_depth_8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

        self.last_layer_depth_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))


    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.up(x_4)
        out_786 = torch.cat((x_3, x_4_), dim=1)
        out_256 = self.conv_256(out_786)
        out_256 = self.ELM_256(out_256)

        branch_16_16 = out_256
        branch_16_8 = F.interpolate(branch_16_16.detach(), scale_factor=2, mode='bilinear', align_corners=False)
        branch_16_4 = F.interpolate(branch_16_8, scale_factor=2, mode='bilinear', align_corners=False)

        x_3_ = self.up(out_256)
        out_384 = torch.cat((x_2, x_3_), dim=1)
        out_128 = self.conv_128(out_384)
        out_128 = self.ELM_128(out_128)
        branch_8_8 = out_128
        branch_8_16 = F.interpolate(branch_8_8.detach(), scale_factor=0.5, mode='bilinear', align_corners=False)
        branch_8_4 = F.interpolate(branch_8_8.detach(), scale_factor=2, mode='bilinear', align_corners=False)

        x_2_ = self.up(out_128)
        out_192 = torch.cat((x_1, x_2_), dim=1)
        out_64 = self.conv_64(out_192)
        out_64 = self.ELM_64(out_64)

        branch_4_4 = out_64
        branch_4_8 = F.interpolate(branch_4_4.detach(), scale_factor=0.5, mode='bilinear', align_corners=False)
        branch_4_16 = F.interpolate(branch_4_8, scale_factor=0.5, mode='bilinear', align_corners=False)

        out = self.up(out_64)
        out = self.up(out)

        branch_16 = torch.cat((branch_16_16, branch_8_16, branch_4_16), dim=1)
        branch_8 = torch.cat((branch_16_8, branch_8_8, branch_4_8), dim=1)
        branch_4 = torch.cat((branch_16_4, branch_8_4, branch_4_4), dim=1)

        branch16 = self.last_layer_depth_16(self.compress_16(branch_16))
        branch8 = self.last_layer_depth_8(self.compress_8(branch_8))
        branch4 = self.last_layer_depth_4(self.compress_4(branch_4))

        return out, branch4, branch8, branch16


class Fenpin(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DF = DF(dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        self.conv_l = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1)
        self.conv_m = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1)
        self.conv_h = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1)
        self.conv_s = nn.Conv2d(dim // 4, dim // 4, kernel_size=3, stride=1, padding=1)
        self.bis1 = nn.Parameter(torch.rand(1, requires_grad=True))
        self.bis2 = nn.Parameter(torch.rand(1, requires_grad=True))
        self.bis3 = nn.Parameter(torch.rand(1, requires_grad=True))
        self.bis4 = nn.Parameter(torch.rand(1, requires_grad=True))

    def forward(self, feature):
        l, m, h, s = self.DF(feature)
        l = self.conv_l(l) * self.bis1
        m = self.conv_m(m) * self.bis2
        h = self.conv_h(h) * self.bis3
        s = self.conv_s(s) * self.bis4
        x = torch.cat((l, m, h, s), dim=1)
        x = self.dwconv(x)
        x = x + feature

        return x


class DF(nn.Module):
    def __init__(self, channel_num):
        super(DF, self).__init__()

        self.conv_offset1 = nn.Conv2d(channel_num, 18, kernel_size=3, stride=1, padding=1)
        self.conv_offset2 = nn.Conv2d(channel_num, 18, kernel_size=3, stride=1, padding=1)
        self.conv_offset3 = nn.Conv2d(channel_num, 18, kernel_size=3, stride=1, padding=1)
        self.conv_offset4 = nn.Conv2d(channel_num, 18, kernel_size=3, stride=1, padding=1)

        self.C0 = DeformConv2d(channel_num, channel_num // 4, kernel_size=3, stride=1, padding=1)
        self.relu0 = nn.LeakyReLU(inplace=True)

        self.C1 = DeformConv2d(channel_num, channel_num // 4, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.C2 = DeformConv2d(channel_num, channel_num // 4, kernel_size=3, stride=1, padding=3, dilation=3)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.C3 = DeformConv2d(channel_num, channel_num // 4, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.R = nn.GELU()


    def forward(self, x):

        x_1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_2 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x_3 = F.interpolate(x_2, scale_factor=2, mode='bilinear', align_corners=False)

        offset_1 = self.conv_offset1(x_1)
        offset = self.conv_offset2(x)
        offset_2 = self.conv_offset3(x_2)
        offset_3 = self.conv_offset4(x_3)

        x_1 = self.C3(x_1, offset_1)
        x_1 = self.relu3(x_1)
        x = self.C2(x, offset)
        x = self.relu2(x)
        x_2 = self.C1(x_2, offset_2)
        x_2 = self.relu1(x_2)
        x_3 = self.C0(x_3, offset_3)
        x_3 = self.relu0(x_3)

        x1 = F.interpolate(x_1, scale_factor=2, mode='bilinear', align_corners=False)  # 0.5--1
        x2 = F.interpolate(x_2, scale_factor=0.5, mode='bilinear', align_corners=False)  # 2--1
        x3 = F.interpolate(x_3, scale_factor=0.25, mode='bilinear', align_corners=False)  # 4--1

        l = self.R(x1)
        m = self.R(x - l)
        h = self.R(x2 - x)
        s = self.R(x3 - x2)

        return l, m, h, s



class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



