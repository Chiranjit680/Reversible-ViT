import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

from tokenmixers import *
from rev import AttentionSubBlock, MLPSubblock

class ViT_OG(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(
            2,
            2,
        ),  # this patch size is used for CIFAR-10
        # --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        enable_amp=False,
        token_mixer="attention",
        pool_size=3,
        num_registers=0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        self.num_registers = num_registers
        self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_registers, embed_dim)) if self.num_registers > 0 else None

        if self.reg_tokens is not None:
            print(f"Initialised register tokens of shape {self.reg_tokens.shape}")

        num_patches = (image_size[0] // self.patch_size[0]) * (
            image_size[1] // self.patch_size[1]
        )

        patches_shape = (image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1])

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible bacpropagation
        # is contrained inside the block code and not exposed.
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                    token_mixer=token_mixer,
                    pool_size=pool_size,
                    patches_shape=patches_shape,
                    num_registers=num_registers
                )
                for _ in range(self.depth)
            ]
        )

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + self.num_registers, self.embed_dim)
        )
        # What kind of a shit initialization is this? Could have used randn * 0.02 like how its done in timm

        # The two streams are concatenated and passed through a linear
        # layer for final projection. This is the only part of RevViT
        # that uses different parameters/FLOPs than a standard ViT model.
        # Note that fusion can be done in several ways, including
        # more expressive methods like in paper, but we use
        # linear layer + LN for simplicity.
        self.head = nn.Linear(self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # patchification using conv and flattening
        # + abolsute positional embeddings
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        if self.num_registers > 0:
            batch_registers = self.reg_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([batch_registers, x], dim=1)

        x += self.pos_embeddings

        # the two streams X_1 and X_2 are initialized identically with x and
        # concatenated along the last dimension to pass into the reversible blocks
        # x = torch.cat([x, x], dim=-1)

        for _, layer in enumerate(self.layers):
            x = layer(x)
        

        # aggregate across sequence length
        x = x.mean(1)

        # head pre-norm
        x = self.norm(x) 

        # pre-softmax logits
        x = self.head(x)

        # return pre-softmax logits
        return x
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, enable_amp, token_mixer, pool_size, patches_shape, num_registers):
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        if token_mixer == "attention":
            self.F = AttentionSubBlock(
                dim=dim, num_heads=num_heads, enable_amp=enable_amp
            )
            print("Using attention token mixer")
        elif token_mixer == "pooling":
            self.F = PoolingFBlock(
                dim=dim, pool_size=pool_size, patches_shape=patches_shape, num_registers=num_registers, enable_amp=enable_amp
            )
            print(f"Using pooling token mixer with pool_size : {pool_size}")
        elif token_mixer == "spatial_mlp":
            self.F = SpatialMLPFBlock(
                dim=dim, patches_shape=patches_shape, enable_amp=enable_amp
            )
            print("Using spatial_mlp token mixer")
        else:
            print(f"Unsupported Token Mixer {token_mixer}")
            quit()

        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

    def forward(self, X):
        X = self.F(X) + X
        X = self.G(X) + X
        return X


