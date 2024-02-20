import logging
from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import roi_align
from .utils import to_2tuple


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

# !
class VisionResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def proj_without_attn(self, value):
        attn_module = self.attn
        value = F.linear(value, attn_module.in_proj_weight,
                         bias=attn_module.in_proj_bias)[..., -attn_module.embed_dim:]
        value = F.linear(value, attn_module.out_proj.weight,
                         bias=attn_module.out_proj.bias)

        return value

    def forward_without_attn(self, q_x):
        x = q_x + self.ls_1(self.proj_without_attn(value=self.ln_1(q_x)))    # use the maskclip-zhou style
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

    # ! 1 더해주는 방식
    # def create_attn_mask(self, num_tokens):
    #     tokens = num_tokens//2   # 0 -> -inf / 1 -> 0 
    #     attn_mask_cls = torch.zeros((tokens, num_tokens))
    #     for i in range(tokens):
    #         attn_mask_cls[i, i] = 1
    #         attn_mask_cls[i, tokens+i] = 1
                
    #     patches = num_tokens//2
    #     attn_mask_image = torch.zeros((patches, num_tokens))
    #     for i in range(patches):
    #         attn_mask_image[i, tokens:] = 1
    #         attn_mask_image[i,i] = 1

    #     attn_mask = torch.cat((attn_mask_cls, attn_mask_image), dim=0)

    #     return attn_mask  

    # ! 제대로된 attn mask 생성
    def create_attn_mask(self, num_tokens, device):
        tokens = num_tokens//2  
        attn_mask_cls = torch.ones((tokens, num_tokens), dtype=torch.bool, device=device)
        for i in range(tokens):
            attn_mask_cls[i, i] = 0
            attn_mask_cls[i, tokens+i] = 0
                    
        patches = num_tokens//2
        attn_mask_image = torch.ones((patches, num_tokens), dtype=torch.bool, device=device)
        for i in range(patches):
            attn_mask_image[i, tokens:] = 0
            attn_mask_image[i,i] = 0

        attn_mask = torch.cat((attn_mask_cls, attn_mask_image), dim=0)

        return attn_mask


    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        # ! attn mask 생성
        if attn_mask is None:
            attn_mask = self.create_attn_mask(q_x.size(0), device=q_x.device)        

        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class ResidualAttentionBlockV2(ResidualAttentionBlock):
    def proj_without_attn(self, value):
        attn_module = self.attn
        value = F.linear(value, attn_module.in_proj_weight,
                         bias=attn_module.in_proj_bias)[..., -attn_module.embed_dim:]
        value = F.linear(value, attn_module.out_proj.weight,
                         bias=attn_module.out_proj.bias)

        return value

    def forward_without_attn(self, q_x):
        x = q_x + self.ls_1(self.proj_without_attn(value=self.ln_1(q_x)))    # use the maskclip-zhou style
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            # ! ResidualAttentionBlockV2 -> ResidualAttentionBlock
            ResidualAttentionBlockV2(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def extract_feature_map(self, x, return_forward=False):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x)
        x_forward = self.resblocks[-1](x)
        x = self.resblocks[-1].forward_without_attn(x)

        if return_forward:
            return x, x_forward
        else:
            return x

    def forward_image_dense(self, x, attn_mask):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x, attn_mask=attn_mask)

        dense = self.resblocks[-1].forward_without_attn(x)
        image = self.resblocks[-1](x, attn_mask=attn_mask)

        return image, dense

# ! image encoder input 달라졌으니 text encoder랑 같은 transformer 쓸 수 없음
class VTransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            VisionResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def extract_feature_map(self, x, return_forward=False):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x)
        x_forward = self.resblocks[-1](x)
        x = self.resblocks[-1].forward_without_attn(x)

        if return_forward:
            return x, x_forward
        else:
            return x

    def forward_image_dense(self, x, attn_mask):
        for i in range(self.layers - 1):
            x = self.resblocks[i](x, attn_mask=attn_mask)

        dense = self.resblocks[-1].forward_without_attn(x)
        image = self.resblocks[-1](x, attn_mask=attn_mask)

        return image, dense


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            input_patchnorm: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            # ! output_tokens: bool = False
            output_tokens: bool = True
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm
        assert not input_patchnorm
        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        # ! 증강된 cls patch 수만큼 PE 차원도 증강
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(4096 * 2, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.transformer = VTransformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.num_heads = heads

        self.global_average_pool = global_average_pool
        # print(f'is attn_pool: {attentional_pool}')
        # attentional_pool: False
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.ln_pre,
                ],
                self.positional_embedding,
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    # self.ln_post,     # fix layer norm
                ],
                # self.proj,        # fix output layers
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def attention_lock(self, **kwargs):
        for name, params in self.named_parameters():
            params.requires_grad = True if "attn" in name or "position" in name else False

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    # ! pooled : CLS token / tokens : patch token
    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        # else:
        #     return x[:, 0], x[:, 1:]
        else:
            cls_tokens = 4096
            return x[:, :cls_tokens], x[:, cls_tokens:]

    def forward(self, x: torch.Tensor):

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        # if self.input_patchnorm:
        #     # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
        #     x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
        #     x = x.permute(0, 2, 4, 1, 3, 5)
        #     x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
        #     x = self.patchnorm_pre_ln(x)
        #     x = self.conv1(x)
        # else:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        # x = torch.cat(
        #     [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #      x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # TODO: Allow interpolating the positional embeddings

        # ! expanded cls token
        expanded_cls_token = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype) # shape = [*, 1, width]
        expanded_cls_token = expanded_cls_token.expand(-1, x.shape[1], -1) # shape = [*, grid**2, width]
        x = torch.cat([expanded_cls_token, x], dim=1)  # shape = [*, (grid ** 2)*2, width]

        # if (h, w) == self.grid_size:
        pe = self.positional_embedding.to(x.dtype)
        # else:
        #     pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled

    def post_attention(self, x):
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled

    def extract_roi_features(self, x, normed_boxes, extract_type='v2'):
        if extract_type == 'v1':
            return self._extract_roi_features_v1(x, normed_boxes)
        elif extract_type == 'v2':
            return self._extract_roi_features_v2(x, normed_boxes)
        else:
            raise NotImplementedError
            # assert extract_type == 'v3'
            # return self._extract_roi_features_v3(x, normed_boxes)

    def mask_pool(self, x, masks):
        feature_map = self.encode_dense(x)
        feature_map = F.normalize(feature_map, dim=-1)

        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).float().flatten(-2, -1)    # bs, h*w
        feature_map = torch.repeat_interleave(
            feature_map, torch.tensor(num_masks_per_image, device=feature_map.device), dim=0)
        features = (feature_map * masks.unsqueeze(-1)).sum(1) / (masks.sum(1, keepdim=True) + 1e-12)

        return features

    def mask_features(self, x, masks):
        feature_map = self.encode_dense(x)
        feature_map = F.normalize(feature_map, dim=-1)

        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).flatten(-2, -1) > 0    # bs, h*w
        feature_map = torch.repeat_interleave(
            feature_map, torch.tensor(num_masks_per_image, device=feature_map.device), dim=0)

        mask_features = [f[m] for m, f in zip(masks, feature_map)]

        return mask_features

    def encode_dense(self, x, keep_shape=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        # assert h == w  # TODO: support input of any shape, need to change the normed boxes to real boxes
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # ! eval에 필요
        expanded_cls_token = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype) # shape = [*, 1, width]
        expanded_cls_token = expanded_cls_token.expand(-1, x.shape[1], -1) # shape = [*, grid**2, width]
        x = torch.cat([expanded_cls_token, x], dim=1)  # shape = [*, (grid ** 2)*2, width]

        pe = self.positional_embedding.to(x.dtype)

        # x = torch.cat(
        #     [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #      x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # if (h, w) == self.grid_size:
        #     pe = self.positional_embedding.to(x.dtype)
        # else:
        #     pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.extract_feature_map(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            # !
            # _, tokens = self._global_pool(x)
            tokens, _ = self._global_pool(x)
        else:
            # !
            # _, tokens = self._global_pool(x)
            tokens, _ = self._global_pool(x)
            tokens = self.ln_post(tokens)

        if self.proj is not None:
            tokens = tokens @ self.proj

        feature_map = tokens.view(bs, h * w, -1)   # .permute(0, 3, 1, 2)
        feature_map = F.normalize(feature_map, dim=-1)   # normalize at the last dimension
        if keep_shape:
            feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        return feature_map

    def mask_crop(self, x, masks):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        num_masks_per_image = [len(masks_per_image) for masks_per_image in masks]
        masks = torch.cat(masks).to(x)    # bs, h, w
        x = torch.repeat_interleave(
            x, torch.tensor(num_masks_per_image, device=x.device), dim=0)
        x = x * masks[:, None]
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # ! eval에 필요
        expanded_cls_token = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype) # shape = [*, 1, width]
        expanded_cls_token = expanded_cls_token.expand(-1, x.shape[1], -1) # shape = [*, grid**2, width]
        x = torch.cat([expanded_cls_token, x], dim=1)  # shape = [*, (grid ** 2)*2, width]

        pe = self.positional_embedding.to(x.dtype)
        # class embeddings and positional embeddings
        # x = torch.cat(
        #     [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #      x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # # TODO: Allow interpolating the positional embeddings

        # if (h, w) == self.grid_size:
        #     pe = self.positional_embedding.to(x.dtype)
        # else:
        #     pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        return pooled

    @staticmethod
    def _generate_masks_per_image(normed_boxes, mask_h, mask_w):
        num_boxes = len(normed_boxes)
        boxes = normed_boxes * torch.tensor(
            [[mask_w, mask_h, mask_w, mask_h]], device=normed_boxes.device)
        masks = torch.zeros(num_boxes, mask_h, mask_w,
                            dtype=torch.bool, device=normed_boxes.device)
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box.long().tolist()
            masks[i, y0:y1, x0:x1] = True

        return masks
    
    @staticmethod
    def _denormalize_boxes(normed_boxes, x):
        h, w = x.shape[-2:]
        denormed_boxes = []
        for boxes in normed_boxes:
            new_boxes = boxes.clone()     # FIXME: do not change the value in normed_boxes!
            new_boxes[:, [0, 2]] *= w
            new_boxes[:, [1, 3]] *= h
            denormed_boxes.append(new_boxes)
        return denormed_boxes

    def _extract_roi_features_v1(self, x, normed_boxes):
        # used masks
        bs, _, h, w = x.shape
        patch_height, patch_width = self.patch_size
        mask_h, mask_w = h // patch_height, w // patch_width
        masks = [self._generate_masks_per_image(normed_boxes_, mask_h, mask_w)
                 for normed_boxes_ in normed_boxes]

        return self.mask_attn_pool(x, masks)

    def _extract_roi_features_v3(self, x, normed_boxes):    # v3 for extract two types
        # used masks
        bs, _, h, w = x.shape
        patch_height, patch_width = self.patch_size
        mask_h, mask_w = h // patch_height, w // patch_width
        masks = [self._generate_masks_per_image(normed_boxes_, mask_h, mask_w)
                 for normed_boxes_ in normed_boxes]

        roi_features_v1, dense_x = self.mask_attn_pool(x, masks, return_dense=True)
        dense_x = F.normalize(dense_x, dim=-1)   # normalize along last dimension
        dense_x = dense_x.permute(0, 3, 1, 2)
        roi_features_v2 = roi_align(dense_x, self._denormalize_boxes(normed_boxes, dense_x), 
                                    (1, 1), 1.0, -1, True)[..., 0, 0]

        return roi_features_v1, roi_features_v2

    def _extract_roi_features_v2(self, x, normed_boxes):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        # assert h == w     # TODO: support input of any shape, need to change the normed boxes to real boxes
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # ! eval에 필요
        expanded_cls_token = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype) # shape = [*, 1, width]
        expanded_cls_token = expanded_cls_token.expand(-1, x.shape[1], -1) # shape = [*, grid**2, width]
        x = torch.cat([expanded_cls_token, x], dim=1)  # shape = [*, (grid ** 2)*2, width]

        pe = self.positional_embedding.to(x.dtype)

        # x = torch.cat(
        #     [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #      x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # if (h, w) == self.grid_size:
        #     pe = self.positional_embedding.to(x.dtype)
        # else:
        #     pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.extract_feature_map(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            # !
            # _, tokens = self._global_pool(x)
            tokens, _ = self._global_pool(x)
        else:
            # !
            # _, tokens = self._global_pool(x)
            tokens, _ = self._global_pool(x)
            tokens = self.ln_post(tokens)

        if self.proj is not None:
            tokens = tokens @ self.proj
        tokens = F.normalize(tokens, dim=-1)   # normalize along last dimension
        tokens = tokens.view(bs, h, w, -1).permute(0, 3, 1, 2)
        return roi_align(tokens, self._denormalize_boxes(normed_boxes, tokens),
                         (1, 1), 1.0, -1, True)[..., 0, 0]

    def rescale_positional_embedding(self, out_size, dtype):
        h, w = out_size
        rescaled_positional_embedding = \
            self.positional_embedding.new_zeros(1 + h*w, self.positional_embedding.shape[1])
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].T.contiguous().view(
            1, -1, *self.grid_size)
        pe_2d = F.interpolate(pe_2d, out_size, mode='bicubic', align_corners=False).view(-1, h*w)
        rescaled_positional_embedding[1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding.to(dtype=dtype)

    def _mask_attn_pool(self, x: torch.Tensor, attn_mask: torch.Tensor, num_mask_tokens: int, return_dense=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        if (h, w) == self.grid_size:
            pe = self.positional_embedding.to(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls_embed = x[0:1]
        cls_embed = cls_embed.expand(num_mask_tokens, -1, -1)
        x = torch.cat([cls_embed, x], dim=0)
        if return_dense:
            x, x_dense = self.transformer.forward_image_dense(x, attn_mask)
            x_dense = x_dense.permute(1, 0, 2)  # LND -> NLD
            x_dense = x_dense[:, num_mask_tokens + 1:]

            x_dense = self.ln_post(x_dense)

            if self.proj is not None:
                x_dense = x_dense @ self.proj
            x_dense = F.normalize(x_dense, dim=-1)  # normalize along last dimension
            x_dense = x_dense.view(bs, h, w, -1)
        else:
            x = self.transformer(x, attn_mask)
            x_dense = None
        x = x.permute(1, 0, 2)  # LND -> NLD

        # [N, L, D]
        x = self.ln_post(x[:, :num_mask_tokens, :])

        if self.proj is not None:
            x = torch.einsum("nld,dc->nlc", x, self.proj)

        return x, x_dense

    def mask_attn_pool(self, image, masks, return_dense=False):
        assert hasattr(self, "positional_embedding")
        batch_size = image.shape[0]
        assert batch_size == len(masks)
        num_masks_per_image = [mask.shape[0] for mask in masks]
        num_queries = max(num_masks_per_image)
        mask_h, mask_w = masks[0].shape[1:]

        batch_masks = torch.ones(batch_size, num_queries, mask_h, mask_w, dtype=torch.bool).to(image.device)
        for batch_id, mask in enumerate(masks):
            batch_masks[batch_id, :mask.shape[0]] = mask

        mask_token_attn_mask = torch.logical_not(batch_masks)
        # [B, Q, H//P x W//P]
        mask_token_attn_mask = mask_token_attn_mask.reshape(batch_size, num_queries, -1)

        num_mask_token = num_queries
        num_image_cls_token = (mask_h * mask_w + 1)
        num_image_token = num_image_cls_token - 1
        num_all_token = num_mask_token + num_image_cls_token

        # we start with no mask out
        attn_mask = torch.zeros(
            (num_all_token, num_all_token), dtype=torch.bool, device=image.device
        )

        # mask+cls+image token to mask token attention is masked out
        attn_mask[:, :num_mask_token] = True

        attn_mask = attn_mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        attn_mask[:, :num_mask_token, -num_image_token:] = mask_token_attn_mask
        num_heads = self.num_heads  # head width 64
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * num_heads, num_all_token, num_all_token)

        batch_mask_features, x_dense = self._mask_attn_pool(image, attn_mask, num_mask_token,
                                                            return_dense=return_dense)

        mask_features = [batch_mask_features[batch_id, :num_masks]
                         for batch_id, num_masks, in enumerate(num_masks_per_image)]
        if return_dense:
            # x_dense = F.normalize(x_dense, dim=-1).flatten(1, 2)   # bs, h*w, c
            # masks = torch.cat(masks).float().flatten(-2, -1)  # bs, h*w
            # x_dense = torch.repeat_interleave(
            #     x_dense, torch.tensor(num_masks_per_image, device=x_dense.device), dim=0)
            # x_dense = (x_dense * masks.unsqueeze(-1)).sum(1) / masks.sum(1, keepdim=True)

            return torch.cat(mask_features), x_dense
        else:
            return torch.cat(mask_features)

    def encode_rois_and_image(self, x, normed_boxes):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        bs, _, h, w = x.shape
        # assert h == w  # TODO: support input of any shape, need to change the normed boxes to real boxes
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # ! eval에 필요
        expanded_cls_token = _expand_token(self.class_embedding, x.shape[0]).to(x.dtype) # shape = [*, 1, width]
        expanded_cls_token = expanded_cls_token.expand(-1, x.shape[1], -1) # shape = [*, grid**2, width]
        x = torch.cat([expanded_cls_token, x], dim=1)  # shape = [*, (grid ** 2)*2, width]

        pe = self.positional_embedding.to(x.dtype)

        # x = torch.cat(
        #     [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
        #      x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # if (h, w) == self.grid_size:
        #     pe = self.positional_embedding.to(x.dtype)
        # else:
        #     pe = self.rescale_positional_embedding(out_size=(h, w), dtype=x.dtype)

        x = x + pe

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, x_image = self.transformer.extract_feature_map(x, return_forward=True)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            # !
            # _, tokens = self._global_pool(x)
            tokens, _ = self._global_pool(x)
        else:
            # !
            # _, tokens = self._global_pool(x)
            tokens, _ = self._global_pool(x)
            tokens = self.ln_post(tokens)

        if self.proj is not None:
            tokens = tokens @ self.proj

        feature_map = tokens.view(bs, h * w, -1)  # .permute(0, 3, 1, 2)
        feature_map = F.normalize(feature_map, dim=-1)
        feature_map = feature_map.view(bs, h, w, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(feature_map, self._denormalize_boxes(normed_boxes, feature_map),
                           (1, 1), 1.0, -1, True)[..., 0, 0]
        x_rois = F.normalize(x_rois, dim=-1)

        x_image = self.post_attention(x_image)
        x_image = F.normalize(x_image, dim=-1)

        return x_rois, x_image


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        assert unlocked_layers == 0 and freeze_layer_norm
        print(f'Freeze the text encoder', flush=True)
        for p in self.parameters():
            p.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
