import torch 
import torch.nn as nn 
import torch.nn.functional as F
from cnn import ConvBlock

# ---------------------------------------------------------
# 1. Patch Embedding: split image into patches + linear proj
# ---------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_ch=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel = patch_size and stride = patch_size
        # acts as patch extract + linear projection
        self.features = nn.Sequential(
            ConvBlock(in_ch, embed_dim//2, embed_dim//2, max_pool=False), 
            ConvBlock(embed_dim//2, embed_dim//2, embed_dim, max_pool=False)
        )
        
        self.project = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.features(x)
        x = self.project(x)          # [B, D, H/P, W/P]
        x = x.flatten(2)             # [B, D, N]
        x = x.transpose(1, 2)        # [B, N, D]   # N = num_patches
        return x


# ---------------------------------------------------------
# 2. Transformer Encoder Block using nn.MultiheadAttention
# ---------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.25):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,  
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # x: [B, N, D]

        # ---- Multi-head Self-Attention ----
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)   
        x = x + self.drop1(attn_out)                      # Residual

        # ---- MLP ----
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)                          # Residual

        return x


# ---------------------------------------------------------
# 3. Complete Vision Transformer (ViT)
# ---------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_ch=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        # ---- Patch embedding ----
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches = self.patch_embed.num_patches

        # ---- CLS token ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Positional embedding ----
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # ---- Transformer encoder blocks ----
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # ---- Final normalization ----
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Classifier ----
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, embed_dim), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        # ---- Patch embedding ----
        x = self.patch_embed(x)      # [B, N, D]

        # ---- Add CLS token ----
        cls_tok = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tok, x], dim=1)          # [B, N+1, D]

        # ---- Add positional embeddings ----
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # ---- Transformer blocks ----
        for blk in self.blocks:
            x = blk(x)

        # ---- Final norm ----
        x = self.norm(x)

        # ---- CLS head ----
        cls_out = x[:, 0]            # take CLS output
        logits = self.head(cls_out)

        return logits
