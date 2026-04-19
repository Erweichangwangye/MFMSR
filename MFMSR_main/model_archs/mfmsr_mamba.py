import math
import numbers

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
#from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from model_archs.utils import selective_scan_flatten, SEModule, x_selective_scan


NEG_INF = -1000000
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq


        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(num_feat//compress_ratio, num_feat // compress_ratio, 3, 1, 1,groups=num_feat//compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(num_feat, num_feat, 3,1,padding=2,groups=num_feat,dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)
        if kwargs.get('sscore_type', 'None') != 'None':
            ms_stage, current_stage = kwargs.get('ms_stage'), kwargs.get('current_layer')
            if current_stage not in ms_stage:
                kwargs['sscore_type'] = 'None'

        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            forward_type = "multiscale_ssm"
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            self.K = 1 + kwargs.get('ms_split')[0]

        self.K2 = self.K if forward_type not in ["share_a"] else 1

        if kwargs.get('add_se', False):
            self.se = SEModule(d_expand, reduction=8)

        if kwargs.get('ms_fusion', None) == None:
            if kwargs.get('upsample', None) == 'interpolate':
                pass
            elif kwargs.get('upsample', None) == 'conv':
                if kwargs.get('current_layer', 0) == 3:
                    self.upsample = nn.ConvTranspose2d(
                        in_channels=d_expand,
                        out_channels=d_expand,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=d_expand,
                        bias=conv_bias,
                    )
                else:
                    self.upsample = nn.ConvTranspose2d(
                        in_channels=d_expand,
                        out_channels=d_expand,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        groups=d_expand,
                        bias=conv_bias,
                    )

        # forward_type =======================================
        self.forward_core = dict(
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            flatten_ssm=self.forward_core_flatten,
            multiscale_ssm=self.forward_core_multiscale,
        ).get(forward_type, self.forward_corev2)

        # in proj =======================================

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )  # branch 0, convert B, C, H, W to B, C, H, W
        if kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            if kwargs.get('add_conv', True):
                b1_stride = 2
                self.conv2d_b1 = nn.Conv2d(
                    in_channels=d_expand,
                    out_channels=d_expand,
                    groups=d_expand,
                    bias=conv_bias,
                    kernel_size=7,
                    stride=b1_stride,
                    padding=3,
                    **factory_kwargs,
                )  # bracnh 1, convert B, C, H, W to B, C, H//4, W//4
            if kwargs.get('sep_norm', False):
                self.out_norm0 = self.out_norm
                self.out_norm1 = nn.LayerNorm(d_inner)

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if self.debug: debug_rec = []
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = x_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            **self.kwargs,
        )
        x, debug_rec = x[0], x[1]
        if self.ssm_low_rank:
            x = self.out_rank(x)
        if self.debug:
            return x, debug_rec
        return x

    def forward_core_flatten(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 2, 1).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = x.transpose(1, 2).contiguous()  # back to (B, L, C)
        x = selective_scan_flatten(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
            **self.kwargs,
        )  # (B, L, C)
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward_core_multiscale(self, xs: list, nrows=-1, channel_first=False):
        nrows = 1
        ys, debug_rec = [], []
        for i, x in enumerate(xs):
            if not channel_first:
                x = x.permute(0, 2, 1).contiguous()
            if self.ssm_low_rank:
                x = self.in_rank(x)
            if self.kwargs.get('sep_norm', False):
                norm_name = getattr(self, "out_norm" + str(i), nn.LayerNorm(self.d_inner))
            else:
                norm_name = getattr(self, "out_norm", None)
            if i == 0:
                proj_weight = self.x_proj_weight[[i]]
                dt_projs_weight = self.dt_projs_weight[[i]]
                dt_projs_bias = self.dt_projs_bias[[i]]
                A_logs = self.A_logs[i * self.d_inner:(i + 1) * self.d_inner]
                Ds = self.Ds[i * self.d_inner:(i + 1) * self.d_inner]
            else:
                proj_weight = self.x_proj_weight[i:]
                dt_projs_weight = self.dt_projs_weight[i:]
                dt_projs_bias = self.dt_projs_bias[i:]
                A_logs = self.A_logs[i * self.d_inner:]
                Ds = self.Ds[i * self.d_inner:]
            # if not debug  mode, remove x_rec
            x, debug = x_selective_scan(
                x, proj_weight, None, dt_projs_weight, dt_projs_bias,
                A_logs, Ds,
                norm_name,
                nrows=nrows, delta_softplus=True, force_fp32=self.training,
                **self.kwargs,
            )

            if self.ssm_low_rank:
                x = self.out_rank(x)  # (B, L, C)
            ys.append(x)
            debug_rec.append(debug)
        if self.debug:
            return ys, debug_rec
        return ys

    def forward(self, x: torch.Tensor, h_tokens=None, w_tokens=None, **kwargs):

        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            b, h, w, d = x.shape
            if not self.disable_z_act:
                z = self.act(z)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x_b1 = x
            x = self.act(self.conv2d(x))  # (b, d, h, w)
            if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
                if self.kwargs.get('add_conv', True):
                    x_b1 = self.act(self.conv2d_b1(x_b1))  # (b, d, h//4, w//4)
                h_b1, w_b1 = x_b1.shape[2:]
                # reverse horizontal scan
                x_hori_r = x_b1.flatten(2).flip(-1)
                # vertical scan
                x_vert = x_b1.transpose(2, 3).flatten(2).contiguous()
                # reverse vertical scan
                x_vert_r = x_b1.transpose(2, 3).flatten(2).flip(-1).contiguous()
                splits = self.kwargs.get('ms_split')[1]
                if splits == 3:
                    x_b1 = torch.cat([x_hori_r, x_vert, x_vert_r], dim=2)  # (b, d, h//2*w//2*3)
                    x = x.flatten(2)  # (b, d, h*w)
                elif splits == 2:
                    x_b1 = torch.cat([x_hori_r, x_vert_r], dim=2)
                elif splits == 1:
                    x_b1 = x_vert_r
                x = [x_b1, x]
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        if self.debug:
            y, debug_rec = self.forward_core(x, channel_first=(self.d_conv > 1))
        else:
            y = self.forward_core(x, channel_first=(self.d_conv > 1))

        if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
            y_b0, y_b1 = y[1], y[0]  # (b, h//4*w//4*3, d)
            if splits == 3:
                y_hori_r = y_b1[:, :h_b1 * w_b1].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
                y_vert = y_b1[:, h_b1 * w_b1:h_b1 * w_b1 * 2].view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1,
                                                                                                              2)
                y_vert_r = y_b1[:, h_b1 * w_b1 * 2:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1,
                                                                                                              2)
                y_b1 = y_hori_r + y_vert + y_vert_r
            elif splits == 2:
                y_hori_r = y_b1[:, :h_b1 * w_b1].flip(-2).view(b, h_b1, w_b1, -1).permute(0, 3, 1, 2)
                y_vert_r = y_b1[:, h_b1 * w_b1:].flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_hori_r + y_vert_r
            elif splits == 1:
                y_vert_r = y_b1.flip(-2).view(b, w_b1, h_b1, -1).transpose(1, 2).permute(0, 3, 1, 2)
                y_b1 = y_vert_r
            y_b1 = F.interpolate(y_b1, size=(h, w), mode='bilinear', align_corners=False)
            y_b1 = y_b1.permute(0, 2, 3, 1).contiguous()
            y = y_b0.view(b, h, w, -1) + y_b1

        if getattr(self, "__DEBUG__", False):
            if self.kwargs.get('sscore_type', 'None') in ['multiscale_4scan_12']:
                ys_b1 = debug_rec[0]['ys'].view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                ys_b1 = F.interpolate(ys_b1, size=(h, w), mode='bilinear', align_corners=False)  # (b*3, d, h, w)
                ys_b1 = ys_b1.view(b, 3, -1, h, w).contiguous()  # (b, 3, d, h, w)
                temp = ys_b1[:, 0].clone()
                ys_b1[:, 0] = ys_b1[:, 1]
                ys_b1[:, 1] = temp
                ys_b0 = debug_rec[1]['ys'].view(b, 1, -1, h, w).contiguous()  # (b, 1, d, h, w)
                ys = torch.cat([ys_b0, ys_b1], dim=1)  # (b, 4, d, h, w)

                xs_b1 = debug_rec[0]['xs'].view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                xs_b1 = F.interpolate(xs_b1, size=(h, w), mode='bilinear', align_corners=False)
                xs_b1 = xs_b1.view(b, 3, -1, h, w).contiguous()
                temp = xs_b1[:, 0].clone()
                xs_b1[:, 0] = xs_b1[:, 1]
                xs_b1[:, 1] = temp
                xs_b0 = debug_rec[1]['xs'].view(b, 1, -1, h, w).contiguous()
                xs = torch.cat([xs_b0, xs_b1], dim=1).view(b, -1, h * w)  # (b, 4, d, h, w)

                A_logs_b1, dts_b1, bs_b1, cs_b1, ds_b1, delta_bias_b1 = debug_rec[0]['As'], debug_rec[0]['dts'], \
                                                                        debug_rec[0]['Bs'], debug_rec[0]['Cs'], \
                                                                        debug_rec[0]['Ds'], debug_rec[0]['delta_bias']
                A_logs_b0, dts_b0, bs_b0, cs_b0, ds_b0, delta_bias_b0 = debug_rec[1]['As'], debug_rec[1]['dts'], \
                                                                        debug_rec[1]['Bs'], debug_rec[1]['Cs'], \
                                                                        debug_rec[1]['Ds'], debug_rec[1]['delta_bias']
                A_logs = torch.cat([A_logs_b0, A_logs_b1.repeat(3, 1)], dim=0)
                delta_bias = torch.cat([delta_bias_b0, delta_bias_b1.repeat(3)], dim=0)
                ds = torch.cat([ds_b0, ds_b1.repeat(3)], dim=0)
                # dts, bs, cs correspond to v,k,q in transformer
                dts_b1 = dts_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                dts_b1 = F.interpolate(dts_b1, size=(h, w), mode='bilinear', align_corners=False)
                dts_b1 = dts_b1.view(b, 3, -1, h * w)
                temp = dts_b1[:, 0].clone()
                dts_b1[:, 0] = dts_b1[:, 1]
                dts_b1[:, 1] = temp

                dts_b1 = dts_b1.view(b, -1, h * w).contiguous()
                dts = torch.cat([dts_b0, dts_b1], dim=1).view(b, -1, h * w)  # (b, 4, d, h*w)

                bs_b1 = bs_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                bs_b1 = F.interpolate(bs_b1, size=(h, w), mode='bilinear', align_corners=False)
                bs_b1 = bs_b1.view(b, 3, -1, h * w)
                temp = bs_b1[:, 0].clone()
                bs_b1[:, 0] = bs_b1[:, 1]
                bs_b1[:, 1] = temp
                bs = torch.cat([bs_b0, bs_b1], dim=1)  # (b, 4, d, h*w)

                cs_b1 = cs_b1.view(b, -1, 3, h_b1, w_b1).permute(0, 2, 1, 3, 4).view(b * 3, -1, h_b1, w_b1)
                cs_b1 = F.interpolate(cs_b1, size=(h, w), mode='bilinear', align_corners=False)
                cs_b1 = cs_b1.view(b, 3, -1, h * w)
                temp = cs_b1[:, 0].clone()
                cs_b1[:, 0] = cs_b1[:, 1]
                cs_b1[:, 1] = temp
                cs = torch.cat([cs_b0, cs_b1], dim=1)  # (b, 4, d, h*w)
            else:
                xs, ys = debug_rec['xs'], debug_rec['ys']
                A_logs, dts, bs, cs, ds, delta_bias = debug_rec['As'], debug_rec['dts'], debug_rec['Bs'], debug_rec[
                    'Cs'], debug_rec['Ds'], debug_rec['delta_bias']

            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=bs, Cs=cs, Ds=ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        if self.kwargs.get('add_se', False):
            y = y.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            y = self.se(y)
            y = y.permute(0, 2, 3, 1).contiguous()
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

# class ModifyPPM(nn.Module):
#     def __init__(self, in_dim, reduction_dim, bins):
#         super(ModifyPPM, self).__init__()
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, bias=False, groups=reduction_dim),
#                 nn.GELU()
#             ))
#         self.features = nn.ModuleList(self.features)
#         self.local_conv = nn.Sequential(
#             nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim),
#             nn.GELU(),
#         )
#
#     def forward(self, x):
#         x_size = x.size()
#         out = [self.local_conv(x)]
#         for f in self.features:
#             out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
#         return torch.cat(out, 1)
#
# class LMSA(nn.Module):
#     def __init__(self, in_dim, hidden_dim, patch_num):
#         super().__init__()
#         self.down_project = nn.Linear(in_dim,hidden_dim)
#         self.act = nn.GELU()
#         self.mppm = ModifyPPM(hidden_dim, hidden_dim //4,  [3,6,9,12])
#         self.patch_num = patch_num
#         self.up_project = nn.Linear(hidden_dim, in_dim)
#         self.down_conv = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),
#                                        nn.GELU())
#
#     def forward(self, x):
#         down_x = self.down_project(x)
#         down_x = self.act(down_x)
#
#         down_x = down_x.permute(0, 3, 1, 2).contiguous()
#         down_x = self.mppm(down_x).contiguous()
#         down_x = self.down_conv(down_x)
#         down_x = down_x.permute(0, 2, 3, 1).contiguous()
#
#         up_x = self.up_project(down_x)
#         return x + up_x
class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        # self.conv_blk = CAB(hidden_dim,is_light_sr)
        self.linear_0 = nn.Conv2d(hidden_dim,hidden_dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        self.dmlp=DMlp(dim=hidden_dim)
        self.conv_blk=MultiFrequencyChannelAttention(
            in_channels=hidden_dim,
            dct_h=7,
            dct_w=7,
            frequency_branches=16,
            frequency_selection='top',
            reduction=16
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))



    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        y,input=self.linear_0(input.permute(0, 3, 1, 2).contiguous()).chunk(2, dim=1)
        input=input.permute(0, 2, 3, 1).contiguous()
        y = self.dmlp(y)
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = self.linear_1(x.permute(0, 3, 1, 2).contiguous() + y).permute(0, 2, 3, 1).contiguous()

        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x
    # def forward(self, input, x_size):
    #     # x [B,HW,C]
    #     B, L, C = input.shape
    #     input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
    #     x = self.ln_1(input)
    #     x = input*self.skip_scale + self.drop_path(self.self_attention(x))
    #     x = x*self.skip_scale2
    #     x = x.view(B, -1, C).contiguous()
    #     return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


#@ARCH_REGISTRY.register()
class MFMSR(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=180,
                 depths=(6, 6, 6, 6,6,6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(MFMSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim


        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr = self.is_light_sr
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr = False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr = is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
