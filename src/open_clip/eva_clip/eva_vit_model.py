# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange

try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_
    
from .transformer import PatchDropout
from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast
from torchvision.ops import roi_align
if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")
from typing import Sequence

#########################################################################
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()

    exp_weight = F.dropout(exp_weight, p = dropout)

    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    scale = None,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8,
    dropout = 0.,
    training = False
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale

    # function

    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index),
                dropout if training else 0.
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim = -1)

        weighted_values = torch.stack(weighted_values, dim = -1)
        exp_weights = torch.stack(exp_weights, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)

###########################################################################################


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        drop=0.,
        subln=False,

        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.xattn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln: 
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
        else: 

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            if attn_mask is not None:
                attn_mask = attn_mask.to(q)
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if self.xattn:
            # q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            # k = k.permute(0, 2, 1, 3)
            # v = v.permute(0, 2, 1, 3)

            #   0.0, 0.125, None
            # x = xops.memory_efficient_attention(
            #     q, k, v,
            #     p=self.xattn_drop,
            #     scale=self.scale,
            #     attn_bias=attn_mask   # to allow masked attention
            #     )

            x = memory_efficient_attention(
                q, k, v,
                dropout=self.xattn_drop_rate,
                scale=self.scale,
                attn_bias=attn_mask,
                training=True
            )
            x = x.permute(0, 2, 1, 3)

            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def proj_without_attn(self, x):
        x = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
        # B, num_heads, C
        x = self.inner_attn_ln(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
                 subln=False, naiveswiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
            xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer,
                subln=subln,
                drop=drop
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    def forward_without_attn(self, x):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn.proj_without_attn(x)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn.proj_without_attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn.proj_without_attn(x)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn.proj_without_attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False,
                 use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False,
                 pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False):
        super().__init__()
        self.image_size = img_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
                # patch_dropout=patch_dropout
            )
        else: 
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        def _unlock(x):
            if isinstance(x, list):
                for g in x:
                    _unlock(g)
            else:
                if isinstance(x, torch.nn.Parameter):
                    x.requires_grad = True
                else:
                    for p in x.parameters():
                        p.requires_grad = True

        for blk in self.blocks[-unlocked_groups:]:
            _unlock(blk)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_all_features=False):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def post_attention(self, x, return_all_features=False):
        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def encode_dense(self, x, keep_shape=True):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[:-1]:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.blocks[-1].forward_without_attn(x)[:, 1:]
        x = self.norm(x)
        x = self.head(x)
        assert self.fc_norm is None

        x = F.normalize(x, dim=-1)   # normalize along last dimension
        if keep_shape:
            x = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
        return x

    def extract_roi_features(self, x, normed_boxes, **kwargs):
        x = self.encode_dense(x, keep_shape=True)

        return roi_align(x, self._denormalize_boxes(normed_boxes, x), (1, 1),
                         1.0, -1, True)[..., 0, 0]

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        if (h, w) == self.patch_embed.patch_shape:
            return self.pos_embed
        rescaled_positional_embedding = \
            self.pos_embed.new_zeros(1, 1 + h*w, self.pos_embed.shape[2])
        rescaled_positional_embedding[0, 0] = self.pos_embed[0, 0]
        pe_2d = self.pos_embed[0, 1:].T.contiguous().view(
            1, -1, *self.patch_embed.patch_shape)
        pe_2d = F.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).view(-1, h*w)
        rescaled_positional_embedding[0, 1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding

    def mask_pool(self, x, masks):
        feature_map = self.encode_dense(x, keep_shape=False)
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).float().flatten(-2, -1)    # bs, h*w
        feature_map = torch.repeat_interleave(
            feature_map, torch.tensor(num_masks_per_image, device=feature_map.device), dim=0)
        features = (feature_map * masks.unsqueeze(-1)).sum(1) / (masks.sum(1, keepdim=True) + 1e-12)

        return features

    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()   # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes

    def encode_rois_and_image(self, x, normed_boxes):
        bs, _, h, w = x.shape
        h = h // self.patch_embed.patch_size[0]
        w = w // self.patch_embed.patch_size[1]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.rescale_positional_embedding(out_size=(h, w))
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[:-1]:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x_image = self.head(
            self.post_attention(
                self.blocks[-1](
                    x, rel_pos_bias=rel_pos_bias)
            )
        )
        x_image = F.normalize(x_image, dim=-1)

        x = self.blocks[-1].forward_without_attn(x)[:, 1:]
        x = self.norm(x)
        x = self.head(x)
        assert self.fc_norm is None
        x = F.normalize(x, dim=-1)   # normalize along last dimension
        x = x.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(x, self._denormalize_boxes(normed_boxes, x),
                           (1, 1), 1.0, -1, True)[..., 0, 0]
        x_rois = F.normalize(x_rois, dim=-1)

        return x_rois, x_image
