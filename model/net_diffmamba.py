import os
import warnings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_

# Decoder
from model.net import Decoder

# DiffMamba
from DiffMamba.block.mamba_block import Spiral_MambaBlock, Zig_MambaBlock, ViM_MambaBlock, VMamba_MambaBlock, EfficientVMamba_MambaBlock
from DiffMamba.tools import spiral, zig, vmamba_

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ModifiedPatchEmbed(nn.Module):
    """
    Patch embedding with ZOE (Zero-Order Embedding) for mask information
    """
    def __init__(self, img_size=224, patch_size=4, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        # Main projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        
        # ZOE: Zero-Order Embedding for mask information
        if mask_chans > 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=patch_size // 2)
            # Initialize ZOE weights to zero
            nn.init.zeros_(self.mask_proj.weight)
            nn.init.zeros_(self.mask_proj.bias)
        else:
            self.mask_proj = None
            
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            # Standard image input
            x = self.proj(x)
            if mask is not None and self.mask_proj is not None:
                mask_feat = self.mask_proj(mask)
                x = x + mask_feat  # ZOE: Add zero-initialized mask features
        else:
            # Already processed tensor from previous stage
            if hasattr(x, 'shape') and len(x.shape) == 4:
                x = self.proj(x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B (H*W) C
        x = self.norm(x)
        
        return x, H, W


class DiffMambaBackbone(nn.Module):
    """
    DiffMamba-based backbone for multi-scale feature extraction
    """
    def __init__(self, input_size=352, in_chans=3, mask_chans=0, block_type='spiral', embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2]):
        super().__init__()
        self.input_size = input_size
        self.mask_chans = mask_chans
        self.embed_dims = embed_dims
        self.depths = depths
        self.block_type = block_type
        
        # time_embed
        self.time_embed = nn.ModuleList([
            TimestepEmbed(embed_dims[i]) for i in range(len(embed_dims))
        ])

        # patch_embed
        self.patch_embed1 = ModifiedPatchEmbed(input_size, patch_size=4, stride=4, in_chans=in_chans, 
                                             embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = ModifiedPatchEmbed(input_size//4, patch_size=2, stride=2, in_chans=embed_dims[0], 
                                             embed_dim=embed_dims[1], mask_chans=0)
        self.patch_embed3 = ModifiedPatchEmbed(input_size//8, patch_size=2, stride=2, in_chans=embed_dims[1], 
                                             embed_dim=embed_dims[2], mask_chans=0)
        self.patch_embed4 = ModifiedPatchEmbed(input_size//16, patch_size=2, stride=2, in_chans=embed_dims[2], 
                                             embed_dim=embed_dims[3], mask_chans=0)
        
        # DiffMamba blocks for each scale
        self.mamba_blocks1 = nn.ModuleList([
            self._make_mamba_block(embed_dims[0], block_type) for _ in range(depths[0])
        ])
        self.mamba_blocks2 = nn.ModuleList([
            self._make_mamba_block(embed_dims[1], block_type) for _ in range(depths[1])
        ])
        self.mamba_blocks3 = nn.ModuleList([
            self._make_mamba_block(embed_dims[2], block_type) for _ in range(depths[2])
        ])
        self.mamba_blocks4 = nn.ModuleList([
            self._make_mamba_block(embed_dims[3], block_type) for _ in range(depths[3])
        ])
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        
        self.initialize_weights()
        
    def _make_mamba_block(self, dim, block_type):
        if block_type == 'spiral':
            matrix_list, original_order_indexes_list = spiral(int(self.input_size / 4))  # Approximate grid size
            return Spiral_MambaBlock(
                token_list=matrix_list[0] if matrix_list else [],
                token_list_reversal=matrix_list[1] if len(matrix_list) > 1 else [],
                origina_list=original_order_indexes_list[0] if original_order_indexes_list else [],
                origina_list_reversal=original_order_indexes_list[1] if len(original_order_indexes_list) > 1 else [],
                D_dim=dim, E_dim=dim*2, dt_rank=16, dim_inner=dim*2, d_state=16, use_mamba2=False
            )
        elif block_type == 'zig':
            token_list, origina_list = zig(int(self.input_size / 4), 0)  # Use index 0
            return Zig_MambaBlock(
                token_list=token_list, origina_list=origina_list,
                D_dim=dim, E_dim=dim*2, dt_rank=16, dim_inner=dim*2, d_state=16, use_mamba2=False
            )
        elif block_type == 'vmamba':
            order_list, original_list = vmamba_(int(self.input_size / 4))
            return VMamba_MambaBlock(
                token_list=order_list, origina_list=original_list,
                D_dim=dim, E_dim=dim*2, dt_rank=16, dim_inner=dim*2, d_state=16, use_mamba2=False
            )
        else:
            return ViM_MambaBlock(
                D_dim=dim, E_dim=dim*2, dt_rank=16, dim_inner=dim*2, d_state=16, use_mamba2=False
            )
    
    def forward(self, x, timesteps, cond_img):
        B = x.shape[0]
        features = []
        
        # Stage 1: 352x352 -> 88x88, dim=64
        x1, H1, W1 = self.patch_embed1(cond_img, x)  # Use cond_img with ZOE
        t_emb1 = self.time_embed[0](timesteps)
        c1 = torch.cat([t_emb1, t_emb1], dim=1)  # (B, 2*D) to match DiffMamba format
        w1 = torch.ones(B, x1.shape[1], 1).to(x1.device)
        
        for block in self.mamba_blocks1:
            x1 = block(x1, c1, w1)
        x1 = self.norm1(x1)
        x1_reshaped = x1.transpose(1, 2).reshape(B, self.embed_dims[0], H1, W1)
        features.append(x1_reshaped)
        
        # Stage 2: 88x88 -> 44x44, dim=128
        x2, H2, W2 = self.patch_embed2(x1_reshaped)
        t_emb2 = self.time_embed[1](timesteps)
        c2 = torch.cat([t_emb2, t_emb2], dim=1)
        w2 = torch.ones(B, x2.shape[1], 1).to(x2.device)
        
        for block in self.mamba_blocks2:
            x2 = block(x2, c2, w2)
        x2 = self.norm2(x2)
        x2_reshaped = x2.transpose(1, 2).reshape(B, self.embed_dims[1], H2, W2)
        features.append(x2_reshaped)
        
        # Stage 3: 44x44 -> 22x22, dim=320
        x3, H3, W3 = self.patch_embed3(x2_reshaped)
        t_emb3 = self.time_embed[2](timesteps)
        c3 = torch.cat([t_emb3, t_emb3], dim=1)
        w3 = torch.ones(B, x3.shape[1], 1).to(x3.device)
        
        for block in self.mamba_blocks3:
            x3 = block(x3, c3, w3)
        x3 = self.norm3(x3)
        x3_reshaped = x3.transpose(1, 2).reshape(B, self.embed_dims[2], H3, W3)
        features.append(x3_reshaped)
        
        # Stage 4: 22x22 -> 11x11, dim=512
        x4, H4, W4 = self.patch_embed4(x3_reshaped)
        t_emb4 = self.time_embed[3](timesteps)
        c4 = torch.cat([t_emb4, t_emb4], dim=1)
        w4 = torch.ones(B, x4.shape[1], 1).to(x4.device)
        
        for block in self.mamba_blocks4:
            x4 = block(x4, c4, w4)
        x4 = self.norm4(x4)
        x4_reshaped = x4.transpose(1, 2).reshape(B, self.embed_dims[3], H4, W4)
        features.append(x4_reshaped)
        
        return features
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias:
                nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)


class net(nn.Module):
    """
    Main network class using DiffMamba backbone for feature extraction
    """
    def __init__(self, class_num=2, mask_chans=0, block_type='spiral', **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        
        # Replace PVT backbone with DiffMamba backbone
        self.backbone = DiffMambaBackbone(
            input_size=352, 
            in_chans=3, 
            mask_chans=mask_chans,
            block_type=block_type,
            embed_dims=[64, 128, 320, 512],
            depths=[2, 2, 4, 2]
        )
        
        # Keep the same decoder
        self.decode_head = Decoder(
            dims=[64, 128, 320, 512], 
            dim=256, 
            class_num=class_num, 
            mask_chans=mask_chans
        )
        
        self._init_weights()
    
    def forward(self, x, timesteps, cond_img):
        # Extract multi-scale features using DiffMamba
        features = self.backbone(x, timesteps, cond_img)
        
        # Process features through decoder
        output, layer1, layer2, layer3, layer4 = self.decode_head(features, timesteps, x)
        return output
    
    def _init_weights(self):
        # Initialize DiffMamba weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
    
    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        return self.forward(x, timesteps, cond_img)
    
    def extract_features(self, cond_img):
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass