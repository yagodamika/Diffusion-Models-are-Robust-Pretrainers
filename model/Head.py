from torch import nn
import torch
from einops import rearrange 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
class LinearHead(nn.Module):
    def __init__(self, args, feature_dim_dict):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((args.pool_size1, args.pool_size2))
        pool_sizes_mult = args.pool_size1 * args.pool_size2
        
        feature_dims = feature_dim_dict[args.blocknum] * pool_sizes_mult
        self.fc = nn.Linear(feature_dims, args.num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

class AttentionHead(nn.Module):
    # mimics class attention fusion
    def __init__(self, args, feature_dim_dict, feature_size_dict, num_heads, mlp_ratio, num_blocks):
        super().__init__()
        bn = args.blocknum 
        feat_size = min(feature_size_dict[bn], args.pre_pool_size)
        attention_dim = feature_dim_dict[bn]
        
        if args.norm_type == "batch":
            norm = nn.BatchNorm2d(feature_dim_dict[bn])
        else:
            norm = nn.LayerNorm([feature_dim_dict[bn], feat_size, feat_size])
                    
        self.pre_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(feat_size),
                norm,
                nn.Conv2d(feature_dim_dict[bn], attention_dim, 1),
                LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')),
            )
        self.intra_inter_block_attention = AttentionLayer(attention_dim, num_heads, mlp_ratio, num_blocks)
        self.feature_dims = attention_dim
        self.head = nn.Linear(self.feature_dims, args.num_classes)
        

    def forward(self, feature):
        x = self.pre_layer(feature)
        x = self.intra_inter_block_attention(x)
        x = self.head(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() # I just default set this to identity function
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

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

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, num_blocks):
        super(AttentionLayer, self).__init__()
        
        # original sizes:
        # dim 1024,    num_heads 8
        # mlp ratio 4, num blocks 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # layer string
        # Use_CLS_Token:True:1024,Insert_CLS_Token,
        # Attention:1024:8:4:2,Extract_CLS_Token
    
        layers = []
        # Insert_CLS_Token
        layers.append(LambdaLayer(lambda x: torch.cat((self.cls_token.to(x.device).expand(x.shape[0], -1, -1),
                                                               x), dim=1)))
        # Attention:1024:8:4:2
        for i in range(num_blocks):
            layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio))
        # Extract_CLS_Token
        layers.append(LambdaLayer(lambda x: x[:, 0]))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x