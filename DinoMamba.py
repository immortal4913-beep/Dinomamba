import torch
import numbers 
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import math
import torch.utils.checkpoint as checkpoint
from thop import profile
import numpy as np
import pywt
import os
from selective_scan_interface import mamba_inner_fn, selective_scan_fn

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
#from mamba_ssm import Mamba
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)

class DinoV2ViTS14(nn.Module):
    def __init__(self,
                 repo_path="./dinov2",
                 ckpt_path="./pretrained/dinov2_vits14_pretrain.pth",
                 device="cuda"):
        super().__init__()
        assert os.path.exists(repo_path), f"repo not found: {repo_path}"
        assert os.path.exists(ckpt_path), f"ckpt not found: {ckpt_path}"

        # 1) 从本地仓库构造模型结构（不联网）
        self.backbone = torch.hub.load(
            repo_path,
            "dinov2_vits14",
            source="local",
            pretrained=False
        )

        # 2) 加载本地预训练权重
        state = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state, strict=True)
        print("🔥Loaded local DINOv2 ViT-S/14 weights!")

        self.backbone.eval().to(device)

        # 3) 默认冻结（提取特征用）
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.device = device

    @torch.no_grad()
    def forward_features(self, x):
        """
        x: [B,3,H,W], arbitrary size
        Return 1-level DINOv2 features
        """

        # DINOv2 requires H/W multiples of 14 (224 recommended)
        x_resized = F.interpolate(
            x, size=(252, 252), mode="bilinear", align_corners=False
        )

        # normalization
        x_norm = (x_resized - 0.5) / 0.5

        feats = self.backbone.get_intermediate_layers(
            x_norm, n=1, reshape=True
        )  # list of 1 [B,384,16,16] etc.

        # Then resize each feature back to match input 256×256
        feat_resized = [
            F.interpolate(f, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            for f in feats
        ]
        return feat_resized

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.body(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        # print(x.shape)
        x = self.body(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj_pre = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1, stride=1)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(self.proj_pre(x)).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


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

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        #self.use_fast_path = use_fast_path
        self.use_fast_path = False

        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w) #B,C,H,W-->B,L,C 传入带偏置的层归一化 再还原回 B,C,H,W

    

class DynamicAdaptiveFusion(nn.Module):

    def __init__(self, dim, window_size=5, reduction_ratio=4):
        super().__init__()
        padding = window_size // 2

        # -----------------------------
        # 1️⃣ 局部空间相关性分支 (Spatial Branch)
        # -----------------------------
        self.spatial_conv = nn.Conv2d(
            dim, dim,
            kernel_size=window_size,
            padding=padding,
            groups=dim,  # depthwise 卷积建模局部结构
            bias=False
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # -----------------------------
        # 2️⃣ 通道全局聚合分支 (Channel Branch)
        # -----------------------------
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid()
        )

        # -----------------------------
        # 3️⃣ 动态卷积核融合 (Dynamic Kernel Aggregation)
        # -----------------------------
        self.kernel_fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

        # -----------------------------
        # 4️⃣ 局部门控 (Local Gating)
        # -----------------------------
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        输入: x -> [B, C, H, W]
        输出: out -> [B, C, H, W]
        """

        # ---- 1️⃣ 局部空间相关性 ----
        spatial_feat = self.spatial_conv(x)
        spatial_weight = self.spatial_attn(spatial_feat)  # [B, C, H, W]

        # ---- 2️⃣ 通道全局权重 ----
        channel_weight = self.channel_fc(x)  # [B, C, 1, 1]
        channel_weight = channel_weight.expand_as(spatial_weight)  # ✅ 广播匹配空间维度

        # ---- 3️⃣ 动态核融合 ----
        dynamic_kernel = torch.cat([spatial_weight, channel_weight], dim=1)
        dynamic_kernel = self.kernel_fuse(dynamic_kernel)
        dynamic_kernel = self.norm(dynamic_kernel)
        dynamic_kernel = self.act(dynamic_kernel)

        # ---- 应用动态核调制 ----
        fused = x * dynamic_kernel

        # ---- 4️⃣ 局部门控增强 ----
        gate = self.gate_gen(fused)  # [B, C, 1, 1]
        out = fused * gate + x * (1 - gate)  # 门控残差增强

        return out



class RMSNorm(nn.Module):
    """ 兼容旧版 PyTorch 的 RMSNorm 实现 """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, ..., C)
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / (x.shape[-1] ** 0.5))
        return x / (norm + self.eps) * self.weight

class HFBranch(nn.Module):
    """
    High-Frequency Compensation Branch (HFBranch)
    - Learnable residual extractor: 3x3 -> GELU -> 3x3
    - Explicit high-pass: hf_hp = hf - blur(hf)  (depthwise blur)
    - Spatial gate: gate = sigmoid(Conv1x1(x))  (per-pixel, per-channel)
    - Output: gated high-frequency residual
    """
    def __init__(self, dim, blur_ks=5):
        super().__init__()
        self.dim = dim

        # 1) Learnable residual extractor
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False,groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False,groups=dim),
        )

        # 2) Depthwise blur (fixed, non-trainable) for high-pass
        # Use AvgPool2d as blur (simple & stable)
        # You can also replace with depthwise Gaussian conv if you want.
        self.blur = nn.AvgPool2d(kernel_size=blur_ks, stride=1, padding=blur_ks // 2)

        # 3) Spatial+channel gate (per-pixel per-channel)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True,groups=dim),
            nn.Sigmoid()
        )

        # 4) Optional normalization (helps stability)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim)  # or LayerNorm2d-like

    def forward(self, x):
        """
        x: [B,C,H,W]
        return: hf_gated [B,C,H,W]
        """
        hf = self.res(x)             # learnable residual
        hf = self.norm(hf)

        hf_lp = self.blur(hf)        # low-pass component
        hf_hp = hf - hf_lp           # explicit high-pass residual

        g = self.gate(x)             # gate computed from original x
        hf_gated = hf_hp * g         # inject only where needed

        return hf_gated

class Decoder(nn.Module):

    def __init__(
        self,
        dim,
        vssm_expansion_factor=1.0,
        bias=False,
        norm_layer: Callable[..., torch.nn.Module] = None,
        patch_px=8,
        eps=1e-6
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = RMSNorm

        self.patch_px = patch_px
        self.eps = eps

        hidden = int(dim * vssm_expansion_factor)

        self.local_fusion = DynamicAdaptiveFusion(dim=dim, window_size=5)


        self.hf_branch = HFBranch(dim)
        self.hf_scale  = nn.Parameter(torch.zeros(1))


        self.mamba1 = Mamba(d_model=hidden)
        self.mamba2 = Mamba(d_model=hidden)

        self.conv_s   = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.conv_out = nn.Conv2d(hidden, dim, 1, bias=bias,groups=dim)

        self.act  = nn.SiLU()
        self.ln_1 = norm_layer(dim)

        self._last_objectness_map = None

    @torch.no_grad()
    def _compute_patch_objectness_pixel(self, dino_feat, H, W):
        """
        dino_feat: [B, Cd, H, W]
        return: objectness_pixel [B, H*W]
        """
        B = dino_feat.shape[0]

        # 1) DINO norm
        norm = torch.norm(dino_feat, dim=1, keepdim=True)

        # 2) normalize to [0,1]
        norm_flat = norm.view(B, -1)
        nmin = norm_flat.min(dim=1, keepdim=True)[0].view(B,1,1,1)
        nmax = norm_flat.max(dim=1, keepdim=True)[0].view(B,1,1,1)
        norm = (norm - nmin) / (nmax - nmin + self.eps)

        # 3) objectness
        obj = 1.0 - norm

        # 4) patch pooling
        factor = max(1, 256 // H)
        k = max(1, self.patch_px // factor)

        pad_h = (k - H % k) % k
        pad_w = (k - W % k) % k
        if pad_h > 0 or pad_w > 0:
            obj = F.pad(obj, (0, pad_w, 0, pad_h), mode="replicate")

        obj_patch = F.avg_pool2d(obj, kernel_size=k, stride=k)
        obj_pixel = F.interpolate(obj_patch, size=(H + pad_h, W + pad_w), mode="nearest")
        obj_pixel = obj_pixel[:, :, :H, :W]

        return obj_pixel.view(B, -1)

    def forward_core(self, x, dino_feat):
        """
        x         : [B, C, H, W]
        dino_feat : [B, Cd, H, W]
        """
        B, C, H, W = x.shape
        D = self.mamba1.d_model
        L = H * W

        # -------------------------------------------------
        # 1) Spatial domain (UNSORTED)
        # -------------------------------------------------
        x_f = self.local_fusion(x)
        hf  = self.hf_branch(x)
        x_f = x_f + self.hf_scale * hf

        # -------------------------------------------------
        # 2) Token construction
        # -------------------------------------------------
        x_m = self.conv_s(x_f)
        xm_mamba, xm_gate = torch.chunk(x_m, 2, dim=1)
        xm_seq = xm_mamba.permute(0, 2, 3, 1).reshape(B, L, D)

        # -------------------------------------------------
        # 3) DINO objectness → sorting
        # -------------------------------------------------
        objectness = self._compute_patch_objectness_pixel(dino_feat, H, W)

        # ✅ NEW: store objectness map (for cross-scale consistency loss)
        self._last_objectness_map = objectness.view(B, 1, H, W)

        idx_fwd = torch.argsort(objectness, dim=1, descending=True)
        idx_bwd = torch.argsort(objectness, dim=1, descending=False)

        inv_idx_fwd = torch.argsort(idx_fwd, dim=1)
        inv_idx_bwd = torch.argsort(idx_bwd, dim=1)

        xm_fwd = torch.gather(xm_seq, 1, idx_fwd.unsqueeze(-1).expand(-1, -1, D))
        xm_bwd = torch.gather(xm_seq, 1, idx_bwd.unsqueeze(-1).expand(-1, -1, D))

        # -------------------------------------------------
        # 4) Bi-Mamba scan
        # -------------------------------------------------
        out_fwd = self.mamba1(xm_fwd)
        out_bwd = self.mamba2(xm_bwd)

        out_fwd = torch.gather(out_fwd, 1, inv_idx_fwd.unsqueeze(-1).expand(-1, -1, D))
        out_bwd = torch.gather(out_bwd, 1, inv_idx_bwd.unsqueeze(-1).expand(-1, -1, D))
        out_seq = 0.5 * (out_fwd + out_bwd)
        #out_seq = out_fwd

        # -------------------------------------------------
        # 5) Restore spatial
        # -------------------------------------------------
        out = out_seq.reshape(B, H, W, D).permute(0, 3, 1, 2)
        out = self.conv_out(self.act(xm_gate) * out)

        return out

    def forward(self, x, dino_feat):
        """
        x         : [B, H, W, C]
        dino_feat : [B, Cd, H, W]
        """
        x_norm = self.ln_1(x)
        x_perm = x_norm.permute(0, 3, 1, 2)

        out = self.forward_core(x_perm, dino_feat)
        out = out + x_perm

        return out.permute(0, 2, 3, 1)


class VSSLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        vssm_expansion_factor=1,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Mamba-based Decoders inside this stage
        self.blocks = nn.ModuleList([
            Decoder(
                dim,
                vssm_expansion_factor,
                bias,
            )
            for _ in range(depth)
        ])

        # Weight init
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        # Stage-level downsample (U-Net encoder)
        if downsample is not None:
            self.downsample = downsample(int(dim), 2, 2)
        else:
            self.downsample = None

    def forward(self, x, dino_feat=None):
        """
        x: [B,H,W,C]
        dino_feat: [B,384,H,W]  (DINO last-layer resized)
        """
        for blk in self.blocks:
            x = blk(x, dino_feat=dino_feat)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class VSSLayer_up(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        vssm_expansion_factor=1,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Create internal Decoder blocks (Mamba-based)
        self.blocks = nn.ModuleList([
            Decoder(
                dim,
                vssm_expansion_factor,
                bias,
            )
            for _ in range(depth)
        ])

        # Initialize important weights
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        # Proper upsampling (input dim, not dim*2)
        if upsample is not None:
            self.upsample = upsample(int(dim * 2), 2, 2)
        else:
            self.upsample = None

    def forward(self, x, dino_feat=None):
        """
        x: [B,H,W,C]
        dino_feat: [B,384,H,W]  (DINO last-layer resized)
        """
        if self.upsample is not None:
            x = self.upsample(x)

        for blk in self.blocks:
            x = blk(x, dino_feat=dino_feat)
        return x


class DinoMamba(nn.Module):
    """
    Pure Mamba Underwater Enhancement Network (SVP REMOVED)
    + Experiment-1 patch-level DINO objectness ordering inside Decoder
    """

    def __init__(self, patch_size=1, in_chans=3,
                 depths=[1, 1, 1, 1],
                 depths_decoder=[1, 1, 1, 1],
                 dims=[32, 64, 128, 256],
                 dims_decoder=[256, 128, 64, 32],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]

        # ================= Patch Embedding =================
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # ================= Stochastic depth =================
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        # ================= Encoder =================
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                VSSLayer(
                    dim=dims[i_layer],
                    depth=depths[i_layer],
                    d_state=d_state,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=DownSample,
                    #downsample=DownSample if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            )

        # ================= Decoder =================
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers_up.append(
                VSSLayer_up(
                    dim=dims_decoder[i_layer],
                    depth=depths_decoder[i_layer],
                    d_state=d_state,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                    norm_layer=norm_layer,
                    upsample=UpSample,
                    #upsample=UpSample if (i_layer != 0) else None,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.final_conv = nn.Conv2d(dims_decoder[-1], 3, 1)

        # ================= Skip Channel Attention =================
        self.skip_attentions = nn.ModuleList(
            [ChannelAttention(dim) for dim in [256, 128, 64]]
        )

        # ================= DINOv2 Backbone =================
        self.dino = DinoV2ViTS14(
            repo_path="./dinov2",
            ckpt_path="./pretrained/dinov2_vits14_pretrain.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ================= Encoder =================
    def forward_features(self, x, dino_last):

        skip_list = []
        feature_list = []

        x = self.patch_embed(x)  # [B,H,W,C]
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)

            # 当前stage分辨率
            Hs, Ws = x.shape[1], x.shape[2]
            dino_stage = F.interpolate(dino_last, size=(Hs, Ws), mode="bilinear", align_corners=False)

            x = layer(x, dino_feat=dino_stage)
            feature_list.append(x)

        return x, skip_list, feature_list

    # ================= Decoder =================
    def forward_features_up(self, x, skip_list, feature_list, dino_last):
        for inx, layer_up in enumerate(self.layers_up):
            # 当前stage分辨率（注意：upsample可能在layer_up内部做）
            Hs, Ws = x.shape[1], x.shape[2]
            dino_stage = F.interpolate(dino_last, size=(Hs, Ws), mode="bilinear", align_corners=False)

            if inx == 0:
                x = layer_up(x, dino_feat=dino_stage)
            else:
                skip = skip_list[-inx]
                skip = skip.permute(0, 3, 1, 2)
                skip = self.skip_attentions[inx - 1](skip)
                skip = skip.permute(0, 2, 3, 1)

                x = layer_up(x + skip, dino_feat=dino_stage)

            feature_list.append(x)

        return x, feature_list

    def forward_final(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.final_conv(x)

    def forward(self, x, return_features=False):
        x_in = x

        # ---- DINO last-layer feature ----
        with torch.no_grad():
            dino_feats = self.dino.forward_features(x_in)  # list of 4
            dino_last  = dino_feats[-1]                    # [B,384,256,256]

        # ---- Encoder & Decoder ----
        x_feat, skip_list, feature_list = self.forward_features(x_in, dino_last)
        x_feat, feature_list = self.forward_features_up(x_feat, skip_list, feature_list, dino_last)

        enhanced_branch = self.forward_final(x_feat)
        enhanced = enhanced_branch + x_in

        if return_features:
            return enhanced, feature_list
        return enhanced


#====================== TEST ======================
if __name__ == "__main__":
    model = DinoMamba().cuda()
    x = torch.ones([1, 3, 256, 256]).cuda()

