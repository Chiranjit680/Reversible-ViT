import torch
from torch import nn
import torch.nn.functional as F
import numpy
# Needed to implement custom backward pass
from torch.autograd import Function as Function
from torch.nn.init import trunc_normal_
# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA
import sys
import numpy as np

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class AsymmetricRevVit(nn.Module):
    def __init__(
        self,
        const_dim=768,
        var_dim=[64, 128, 320, 512],
        sra_R=[8, 4, 2, 1],
        n_head=8,
        stages=[3, 3, 6, 3],
        drop_path_rate=0,
        patch_size=(
            2,
            2,
        ),  
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        enable_amp=False,
    ):
        super().__init__()

        self.const_dim = const_dim
        self.n_head = n_head
        self.patch_size = patch_size

        self.const_num_patches = (image_size[0] // self.patch_size[0]) * (
            image_size[1] // self.patch_size[1]
        )

        const_patches_shape = (image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1])

        blk_idx_temp = 0
        block_to_stage_indexing = {}
        for stg_idx_temp, num_blocks in enumerate(stages):
            for _ in range(num_blocks):
                block_to_stage_indexing[blk_idx_temp] = stg_idx_temp
                blk_idx_temp += 1

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(stages))
        ] # stochastic depth decay rule

        # R = []
        # var_patches_shape = []
        # self.var_dim = []
        # const_var_token_ratio = []
        # for i in range(len(stages)):
        #     for _ in range(stages[i]):
        #         R.append(sra_R[i])
        #         var_patches_shape.append((const_patches_shape[0] // 2**i, const_patches_shape[1] // 2**i))
        #         self.var_dim.append(var_dim[i])
        #         const_var_token_ratio.append(2**(2*i))
                

        assert const_patches_shape[0] % 2**(len(stages)-1) == 0
        assert const_patches_shape[1] % 2**(len(stages)-1) == 0

        self.layers = []
        for i in range(sum(stages)):
            stage_index = block_to_stage_indexing[i]

            self.layers.append(
                AsymmetricReversibleBlock(
                    dim_c=self.const_dim,
                    dim_v=var_dim[stage_index], # Same dim_v used for all blocks in a stage
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                    sr_ratio=sra_R[stage_index], # Same sr_ratio used for all blocks in a stage
                    token_map_pool_size=2**stage_index, # Same N_c : N_v ratio used for all blocks in a stage
                    drop_path=dpr[i],                   # Drop path rate depends on block #, not stage #
                    const_patches_shape=const_patches_shape,
                    block_id=i
                )
            )
            
            # Stage transitions
            if (i == np.cumsum(stages)[stage_index] - 1) and (stage_index != len(stages) - 1):
                self.layers.append(
                    VarStreamDownSamplingBlock(
                        input_patches_shape=(const_patches_shape[0] // 2**stage_index, const_patches_shape[1] // 2**stage_index),
                        kernel_size=2, 
                        dim_in=var_dim[stage_index], 
                        dim_out=var_dim[stage_index+1]
                    )
                )

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible bacpropagation
        # is contrained inside the block code and not exposed.
        self.layers = nn.ModuleList(self.layers)

        # Boolean to switch between vanilla backprop and
        # rev backprop. See, ``--vanilla_bp`` flag in main.py
        self.no_custom_backward = False

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed2 = nn.Conv2d(
            3, self.const_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings2 = nn.Parameter(
            torch.zeros(1, self.const_num_patches, self.const_dim)
        )

        self.patch_embed1 = nn.Conv2d(
            3, var_dim[0], kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings1 = nn.Parameter(
            torch.zeros(1, self.const_num_patches, var_dim[0])
        )

        # The two streams are concatenated and passed through a linear
        # layer for final projection. This is the only part of RevViT
        # that uses different parameters/FLOPs than a standard ViT model.
        # Note that fusion can be done in several ways, including
        # more expressive methods like in paper, but we use
        # linear layer + LN for simplicity.
        self.head = nn.Linear(self.const_dim, num_classes, bias=True) # Class Prediction using Const Stream
        self.norm = nn.LayerNorm(self.const_dim) # Class Prediction using Const Stream

    @staticmethod
    def vanilla_backward(x1, x2, layers):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """

        for _, layer in enumerate(layers):
            x1, x2 = layer(x1, x2)

        return x1, x2

    def forward(self, x):
        # patchification using conv and flattening
        # + abolsute positional embeddings
        x2 = self.patch_embed2(x).flatten(2).transpose(1, 2)
        x2 += self.pos_embeddings2

        x1 = self.patch_embed1(x).flatten(2).transpose(1, 2)
        x1 += self.pos_embeddings1

        # the two streams X_1 and X_2 are initialized identically with x and
        # concatenated along the last dimension to pass into the reversible blocks
        # x = torch.cat([x1, x], dim=-1)

        reversible_segments = [[0]]
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], VarStreamDownSamplingBlock):
                reversible_segments[-1].append(i)
                reversible_segments.append([i+1])
        reversible_segments[-1].append(len(self.layers))

        for segment in reversible_segments:

            # no need for custom backprop in eval/inference phase
            if not self.training or self.no_custom_backward:
                executing_fn = AsymmetricRevVit.vanilla_backward
            else:
                executing_fn = RevBackProp.apply

            # This takes care of switching between vanilla backprop and rev backprop
            x1, x2 = executing_fn(
                x1, x2,
                self.layers[segment[0]:segment[1]],
            )

            if segment[1] != len(self.layers):
                x1, x2 = self.layers[segment[1]](x1, x2)

        # aggregate across sequence length
        pred = x2.mean(1) # Class Prediction using Const Stream

        # head pre-norm
        pred = self.norm(pred)

        # pre-softmax logits
        pred = self.head(pred)

        # return pre-softmax logits
        return pred


class RevBackProp(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        X_1, X_2,
        layers,
    ):
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        """

        # obtaining X_1 and X_2 from the concatenated input
        # X_1, X_2 = torch.chunk(x, 2, dim=-1)
        # X_1, X_2 = torch.split(x, [x.size(1) - const_num_patches, const_num_patches], dim=1)

        for layer in layers:
            X_1, X_2 = layer(X_1, X_2)
            all_tensors = [X_1.detach(), X_2.detach()]

        # saving only the final activations of the last reversible block
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return X_1, X_2

    @staticmethod
    def backward(ctx, dX_1, dX_2):
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        # dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve the last saved activations, to start rev recomputation
        X_1, X_2 = ctx.saved_tensors
        # layer weights
        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            # this is recomputing both the activations and the gradients wrt
            # those activations.
            X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
            )
        # final input gradient to be passed backward to the patchification layer
        # dx = torch.cat([dX_1, dX_2], dim=-1)

        # del dX_1, dX_2, X_1, X_2

        return dX_1, dX_2, None, None


class AsymmetricReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self, dim_c, dim_v, num_heads, enable_amp, drop_path, token_map_pool_size, sr_ratio, const_patches_shape, block_id, token_mixer="spatial_mlp"):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.

        self.drop_path_rate = drop_path

        # self.F = SRASubBlockC2V(
        #     dim_c=dim_c,
        #     dim_v=dim_v,
        #     num_heads=num_heads,
        #     patches_shape=const_patches_shape,
        #     token_pool_size=token_map_pool_size,
        #     sr_ratio=sr_ratio, 
        #     enable_amp=enable_amp
        # )

        # self.F = TokenMixerFBlockC2V(
        #     dim_c=dim_c,
        #     dim_v=dim_v,
        #     patches_shape=const_patches_shape,
        #     token_pool_size=token_map_pool_size,
        #     enable_amp=enable_amp
        # )
        # self.F=ResMLP(
        #     dim_c=dim_c,
        #     dim_v=dim_v,
        #     patches_shape=const_patches_shape,
        #     token_pool_size=token_map_pool_size,
        #     enable_amp=enable_amp
            
        # )
        window_size=7
        shift_size=window_size // 2
        # self.F = ShiftedWindowAttentionC2V(
        #     dim_c=dim_c,
        #     dim_v=dim_v,
        #     patches_shape=const_patches_shape,
        #     token_pool_size=token_map_pool_size,
        #     num_heads=num_heads,
        #     window_size=window_size,
        #     shift_size=shift_size,
        #     enable_amp=enable_amp
        # )
        
        
    #     self.F = MultiScaleBlock(
    #     dim_c=dim_c,
    #     dim_v=dim_v,
    #     num_heads=num_heads,
    #     hw_shape=const_patches_shape,  # CIFAR 10   
    #     patches_shape=const_patches_shape,  # Provide the patches shape
    #     qkv_bias=True,
    #     drop_path=drop_path,
    #     norm_layer=nn.LayerNorm,
    #     kernel_q=(token_map_pool_size, token_map_pool_size),  # Use token_map_pool_size
    #     kernel_kv=(token_map_pool_size, token_map_pool_size),  # Use token_map_pool_size
    #     stride_q=(token_map_pool_size, token_map_pool_size),  # Use token_map_pool_size
    #     stride_kv=(token_map_pool_size, token_map_pool_size),  # Use token_map_pool_size
    #     mode="conv",
    #     has_cls_embed=False,
    #     pool_first=False,
    #     rel_pos_spatial=True,
    #     rel_pos_zero_init=False,
    #     residual_pooling=True,
    #     dim_mul_in_att=False
    # )
        self.F = MultiScaleBlock(
            dim_c=dim_c,
            dim_v=dim_v,
            patches_shape=const_patches_shape,
            hw_shape=const_patches_shape,
            num_heads=num_heads,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            kernel_q=token_map_pool_size,  # Single integer
            kernel_kv=token_map_pool_size,  # Single integer
            stride_q=token_map_pool_size,
            stride_kv=token_map_pool_size,
            mode="conv",
            has_cls_embed=False,
            pool_first=False,
            rel_pos_spatial=True,
            rel_pos_zero_init=False,
            residual_pooling=True,
            dim_mul_in_att=False
        )

        self.G = MLPSubblockV2C(
            dim_c=dim_c,
            dim_v=dim_v,
            const_patches_shape=const_patches_shape,
            token_pool_size=token_map_pool_size,
            enable_amp=False,  # standard for ViTs
        )

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

        self.seeds = {}
        self.block_id = block_id
        print(f"Block index : {self.block_id} | Dpr : {self.drop_path_rate}")
    
    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        self.seed_cuda("attn")
        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        self.seed_cuda("droppath")
        f_X_2_dropped = drop_path(
            f_X_2, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2_dropped

        # free memory since X_1 is now not needed
        del X_1

        self.seed_cuda("FFN")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        g_Y_1_dropped = drop_path(
            g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1_dropped

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2  # Add this return statement

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention

        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["FFN"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = drop_path(
                g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = drop_path(
                f_X_2, drop_prob=self.drop_path_rate, training=self.training
            )

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2


class MLPSubblockV2C(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        const_patches_shape,
        token_pool_size,
        enable_amp=False,  # standard for ViTs
    ):
        super().__init__()

        self.patches_shape = (const_patches_shape[0]//token_pool_size, const_patches_shape[1]//token_pool_size)
        self.dim_c = dim_c

        self.norm = nn.LayerNorm(dim_v)

        self.fc1 = nn.Linear(dim_v, dim_c)
        self.act = nn.GELU()
        self.convtranspose = nn.ConvTranspose2d(in_channels=dim_c, out_channels=dim_c, 
                                                kernel_size=token_pool_size, stride=token_pool_size, groups=1)

        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)
            B, N, d_v = x.shape
            x = self.act(self.fc1(x))
            x = x.transpose(1,2).reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])

            x = self.convtranspose(x)
            x = x.reshape(B, self.dim_c, -1).transpose(1, 2)
            return x
            

class SRASubBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.

    Attention with Spatial-Reduction

    F : (N_c, d_c) --> (N_v, d_v)
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        num_heads,
        patches_shape,
        token_pool_size,
        sr_ratio,
        enable_amp=False,
        qk_scale=None,
        qkv_bias=True,

    ):
        super().__init__()
        
        assert dim_v % num_heads == 0
        assert patches_shape[0] % token_pool_size == 0
        assert patches_shape[1] % token_pool_size == 0

        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.num_heads = num_heads
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)
        head_dim = dim_v // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.pool = nn.AvgPool2d(token_pool_size, stride=token_pool_size)

        self.q = nn.Linear(dim_c, dim_v, bias=qkv_bias)
        self.kv = nn.Linear(dim_c, dim_v * 2, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_v, dim_v)
        # self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio # R --> effective receptive field size patched together
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim_c, dim_c, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim_c)

        self.enable_amp = enable_amp

    def forward(self, x):
        # See MLP fwd pass for explanation.
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)

            B, N, d_c = x.shape

            x = x.transpose(1,2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.pool(x)
            x = x.reshape(B, self.N_v, self.dim_c)

            q = self.q(x).reshape(B, self.N_v, self.num_heads, self.dim_v // self.num_heads).permute(0, 2, 1, 3) # bs x num_heads x N x C/num_heads 

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, self.dim_c, self.patches_shape[0]//self.token_pool_size, self.patches_shape[1]//self.token_pool_size)
                x_ = self.sr(x_).reshape(B, self.dim_c, -1).permute(0, 2, 1) # bs x (H/R*W/R) x C
                x_ = self.sr_norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.dim_v // self.num_heads).permute(2, 0, 3, 1, 4) # 2 x bs x num_heads x (H/R*W/R) x C/num_heads
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.dim_v // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1] # bs x num_heads x (H/R*W/R) x C/num_heads

            attn = (q @ k.transpose(-2, -1)) * self.scale # bs x num_heads x N x (H/R*W/R)
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, self.N_v, self.dim_v) # bs x num_heads x N x C/num_heads --> bs x N x C
            x = self.proj(x)
            # x = self.proj_drop(x)

            return x


class TokenMixerFBlockC2V(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim_c,
        dim_v,
        patches_shape,
        token_pool_size,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)

        self.dim_c, self.dim_v = dim_c, dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.N_v = (self.patches_shape[0]//token_pool_size) * (self.patches_shape[1]//token_pool_size)

        # self.pool = nn.AvgPool2d(token_pool_size, stride=token_pool_size)
        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, kernel_size=token_pool_size, stride=token_pool_size)
        self.token_mixer = nn.Linear(self.N_v, self.N_v)

        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            B, N, d_c = x.shape
            
            x = self.norm(x)    

            x = x.transpose(1,2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.conv(x).reshape(B, self.dim_v, self.N_v)

            x = self.token_mixer(x).transpose(1, 2)
            return x


class AffineLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.weight + self.bias


class ResMLP(nn.Module):
    def __init__(self, dim_c, dim_v, patches_shape, token_pool_size, hidden_dim=None, enable_amp=False):
        super().__init__()
        self.dim_c = dim_c  # Input channel dimension
        self.dim_v = dim_v  # Output channel dimension
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.enable_amp = enable_amp
        
        # Calculate number of patches before and after pooling
        self.N = patches_shape[0] * patches_shape[1]  # Original number of patches
        self.N_v = (patches_shape[0] // token_pool_size) * (patches_shape[1] // token_pool_size)  # After pooling
        
        if hidden_dim is None:
            hidden_dim = dim_v * 4  # Default expansion ratio
        
        # Initial normalization and spatial/channel transformation
        self.norm_input = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)
        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, 
                             kernel_size=token_pool_size, stride=token_pool_size)
        
        # Cross-patch sublayer components (operates on N_v patches)
        self.norm1 = AffineLayer(dim_v)
        self.cross_patch_linear = nn.Linear(self.N_v, self.N_v)
        
        # Cross-channel sublayer components  
        self.norm2 = AffineLayer(dim_v)
        self.cross_channel_mlp = nn.Sequential(
            nn.Linear(dim_v, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_v),
        )
        self.norm3 = AffineLayer(dim_v)

    def forward(self, x):
        # Input shape: [B, N, dim_c] e.g., [batch, 3136, 192]
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            B, N, d_c = x.shape
            
            # Initial transformation (matching TokenMixerFBlockC2V)
            x = self.norm_input(x)  # LayerNorm on input
            x = x.transpose(1, 2)   # [B, dim_c, N]
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])  # [B, dim_c, H, W]
            x = self.conv(x).reshape(B, self.dim_v, self.N_v)  # [B, dim_v, N_v]
            x = x.transpose(1, 2)   # [B, N_v, dim_v]
            
            # Now x has shape [B, N_v, dim_v] - this becomes our working dimension
            
            # Cross-patch sublayer (token mixing across N_v patches)
            residual1 = x
            x = self.norm1(x)                    # [B, N_v, dim_v]
            x = x.transpose(1, 2)                # [B, dim_v, N_v]
            x = self.cross_patch_linear(x)       # Mix across N_v patches
            x = x.transpose(1, 2)                # [B, N_v, dim_v]
            x = x + residual1                    # Skip connection
            
            # Cross-channel sublayer (channel mixing)
            residual2 = x
            x = self.norm2(x)                    # [B, N_v, dim_v]
            x = self.cross_channel_mlp(x)        # Mix channels
            x = self.norm3(x)                    # Final normalization
            x = x + residual2                    # Skip connection
            
            # Output shape: [B, N_v, dim_v] e.g., [batch, 784, 64]
            return x
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftedWindowAttentionC2V(nn.Module):
    """
    Shifted Window Attention block for C2V transformation in reversible blocks.
    Adapts Swin Transformer attention for asymmetric vision transformer architecture.
    """
    
    def __init__(
        self,
        dim_c,
        dim_v,
        patches_shape,
        token_pool_size,
        num_heads=8,
        window_size=7,
        shift_size=None,
        enable_amp=False,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        
        self.dim_c = dim_c
        self.dim_v = dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.enable_amp = enable_amp
        
        # Calculate target resolution after pooling
        self.target_H = patches_shape[0] // token_pool_size
        self.target_W = patches_shape[1] // token_pool_size
        self.N_v = self.target_H * self.target_W
        
        # Adjust window size if necessary
        min_dim = min(self.target_H, self.target_W)
        self.window_size = min(window_size, min_dim)
        
        # Set shift size (typically half of window size for shifted attention)
        if shift_size is None:
            self.shift_size = self.window_size // 2
        else:
            self.shift_size = min(shift_size, self.window_size // 2)
        
        # Input normalization and spatial transformation
        self.norm_input = nn.LayerNorm(dim_c, eps=1e-6, elementwise_affine=True)
        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, 
                             kernel_size=token_pool_size, stride=token_pool_size)
        
        # Window attention module
        self.attn = WindowAttention(
            dim=dim_v,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # Normalization for attention
        self.norm_attn = nn.LayerNorm(dim_v)
        
        # Create attention mask for shifted window attention if needed
        if self.shift_size > 0:
            self._create_attention_mask()
        else:
            self.register_buffer("attn_mask", None)
    
    def _create_attention_mask(self):
        """Create attention mask for shifted window attention."""
        H, W = self.target_H, self.target_W
        img_mask = torch.zeros((1, H, W, 1))
        
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, dim_c] where N = patches_shape[0] * patches_shape[1]
        Returns:
            Output tensor of shape [B, N_v, dim_v] where N_v = (H//pool_size) * (W//pool_size)
        """
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            B, N, d_c = x.shape
            
            # Initial transformation (similar to TokenMixerFBlockC2V and ResMLP)
            x = self.norm_input(x)
            x = x.transpose(1, 2)  # [B, dim_c, N]
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.conv(x).reshape(B, self.dim_v, self.N_v)  # [B, dim_v, N_v]
            x = x.transpose(1, 2)  # [B, N_v, dim_v]
            
            # Prepare for window attention
            H, W = self.target_H, self.target_W
            assert x.shape[1] == H * W, f"Expected {H * W} tokens, got {x.shape[1]}"
            
            # Apply layer norm before attention
            shortcut = x
            x = self.norm_attn(x)
            x = x.view(B, H, W, self.dim_v)
            
            # Apply cyclic shift if needed
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x
            
            # Partition into windows
            x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, window_size, window_size, dim_v]
            x_windows = x_windows.view(-1, self.window_size * self.window_size, self.dim_v)  # [nW*B, Wh*Ww, dim_v]
            
            # Apply window attention
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [nW*B, Wh*Ww, dim_v]
            
            # Merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim_v)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, dim_v]
            
            # Reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            
            x = x.view(B, H * W, self.dim_v)  # [B, N_v, dim_v]
            
            # Add residual connection
            x = shortcut + x
            
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
class SwinMLPBlock(nn.Module):
    def __init__(
        self,
        dim_c,
        dim_v,
        patches_shape,
        token_pool_size,
        hidden_dim=None,
        enable_amp=False,
        window_size=7,
        shift_size=0,
        activation=nn.GELU,
        mlp_ratio=4.0,
        num_heads=8,
        drop_path_rate=0.1,
        dropout=0.0,
    ):
        super().__init__()
        self.dim_c = dim_c
        self.dim_v = dim_v
        self.patches_shape = patches_shape
        self.token_pool_size = token_pool_size
        self.hidden_dim = hidden_dim
        self.enable_amp = enable_amp
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.activation = activation
        self.drop_path_rate = drop_path_rate
        
        # Target patch shape after pooling
        self.target_H = patches_shape[0] // token_pool_size
        self.target_W = patches_shape[1] // token_pool_size
        
        if min(self.patches_shape) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.patches_shape)
        assert 0 <= self.shift_size < self.window_size
        self.padding = [
            self.window_size - self.shift_size,
            self.shift_size,
            self.window_size - self.shift_size,
            self.shift_size,
        ]

        # Input normalization and spatial transformation
        self.norm_input = nn.LayerNorm(dim_c, eps=1e-6)
        self.conv = nn.Conv2d(in_channels=dim_c, out_channels=dim_v, 
                             kernel_size=token_pool_size, stride=token_pool_size)
        
        # Window MLP components
        self.norm_spatial = nn.LayerNorm(dim_v)
        self.spatial_mlp = nn.Conv1d(
            self.num_heads * self.window_size**2,
            self.num_heads * self.window_size**2,
            kernel_size=1,
            groups=self.num_heads,
        )
        
        # Channel MLP components
        self.norm_channel = nn.LayerNorm(dim_v)
        self.mlp = Mlp(
            in_features=dim_v,
            hidden_features=int(dim_v * mlp_ratio),
            out_features=dim_v,
            act_layer=activation,
            drop=dropout,
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            B, N, C = x.shape
            
            # Initial transformation: C dimension and pooling
            x = self.norm_input(x)
            x = x.transpose(1, 2)
            x = x.reshape(B, self.dim_c, self.patches_shape[0], self.patches_shape[1])
            x = self.conv(x)
            x = x.reshape(B, self.dim_v, self.target_H * self.target_W).transpose(1, 2)
            
            # Store for residual connection
            residual = x
            
            # Apply spatial MLP with windows
            x = self.norm_spatial(x)
            x = x.view(B, self.target_H, self.target_W, self.dim_v)
            
            # Apply shift if needed
            P_l, P_r, P_t, P_b = self.padding
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_padded = F.pad(shifted_x, [0, 0, P_l, P_r, P_t, P_b])
            else:
                x_padded = F.pad(x, [0, 0, P_l, P_r, P_t, P_b])
            
            # Window partitioning
            x_windows = window_partition(x_padded, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, self.dim_v)
            
            # Reshape for Conv1d spatial mixing
            x_windows_heads = x_windows.view(
                -1, self.window_size * self.window_size, self.num_heads, self.dim_v // self.num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)
            x_windows_heads = x_windows_heads.reshape(
                -1, self.num_heads * self.window_size**2, self.dim_v // self.num_heads)
            
            # Apply spatial mixing
            spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
            
            # Reshape back
            spatial_mlp_windows = spatial_mlp_windows.view(
                -1, self.num_heads, self.window_size**2, self.dim_v // self.num_heads).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size**2, self.dim_v)
            
            # Window reverse
            spatial_mlp_windows = spatial_mlp_windows.view(
                -1, self.window_size, self.window_size, self.dim_v)
            x_unwindowed = window_reverse(spatial_mlp_windows, self.window_size, 
                                         x_padded.shape[1], x_padded.shape[2])
            
            # Remove padding and reverse shift if needed
            if self.shift_size > 0:
                x_unshifted = x_unwindowed[:, P_t:-P_b, P_l:-P_r, :] if P_b > 0 else x_unwindowed[:, P_t:, P_l:-P_r, :]
                x = torch.roll(x_unshifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = x_unwindowed[:, P_t:-P_b, P_l:-P_r, :] if P_b > 0 else x_unwindowed[:, P_t:, P_l:-P_r, :]
            
            # Flatten spatial dimensions
            x = x.view(B, self.target_H * self.target_W, self.dim_v)
            
            # Add residual with drop path
            x = residual + drop_path(x, drop_prob=self.drop_path_rate, training=self.training)
            
            # Apply channel MLP
            residual2 = x
            x = self.norm_channel(x)
            x = self.mlp(x)
            x = residual2 + drop_path(x, drop_prob=self.drop_path_rate, training=self.training)
            
            return x

#Helper functions for MHPA..............
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    """Apply pooling to attention tensor while preserving class token if present."""
    if pool is None:
        return tensor, hw_shape
    
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    H, W = hw_shape
    assert H * W == L, f"Height*Width ({H}*{W}={H*W}) must equal sequence length ({L})"
    
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()
    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    
    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    
    return tensor, hw_shape


def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    """Apply spatial relative positional embeddings to attention."""
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h, device=q.device)[:, None] * q_h_ratio - 
        torch.arange(k_h, device=q.device)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w, device=q.device)[:, None] * q_w_ratio - 
        torch.arange(k_w, device=q.device)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)



def _to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple) and len(x) == 1:
        return (x[0], x[0])
    return x

class MHPA(nn.Module):
    """Multi-Head Pooled Attention with various pooling strategies."""
    
    def __init__(
        self,
        dim_c,
        dim_v,
        patches_shape,
        num_heads=8,
        qkv_bias=False,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        mode="conv",
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
    ):
        super().__init__()
        
        # Basic parameters
        self.pool_first = pool_first
        self.num_heads = num_heads
        self.dim_out = dim_v
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        self.residual_pooling = residual_pooling
        
        head_dim = dim_v // num_heads
        self.scale = head_dim ** -0.5

        if isinstance(kernel_q, int):
          kernel_q = (kernel_q, kernel_q)
        if isinstance(kernel_kv, int):
            kernel_kv = (kernel_kv, kernel_kv)
        
        # Padding calculations
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]
        
        # QKV projections
        if pool_first:
            self.q = nn.Linear(dim_c, dim_v, bias=qkv_bias)
            self.k = nn.Linear(dim_c, dim_v, bias=qkv_bias)
            self.v = nn.Linear(dim_c, dim_v, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim_c, dim_v * 3, bias=qkv_bias)
        


        # Skip pooling with kernel and stride size of (1, 1)
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        # Pooling layers
        self._setup_pooling_layers(
            mode, kernel_q, kernel_kv, stride_q, stride_kv, 
            padding_q, padding_kv, dim_c, dim_v, num_heads, 
            pool_first, norm_layer
        )
        
        # Relative positional embeddings
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            self._setup_relative_position_embeddings(
                patches_shape, stride_q, stride_kv, head_dim, rel_pos_zero_init
            )


    def _setup_pooling_layers(self, mode, kernel_q, kernel_kv, stride_q, stride_kv,
                            padding_q, padding_kv, dim_c, dim_v, num_heads, 
                            pool_first, norm_layer):
        """Setup pooling layers based on the specified mode."""
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0 else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0 else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0 else None
            )
        elif mode in ("conv", "conv_unshared"):
            if pool_first:
                dim_conv = dim_c // num_heads if mode == "conv" else dim_c
            else:
                dim_conv = dim_v // num_heads if mode == "conv" else dim_v
                
            self.pool_q = (
                nn.Conv2d(dim_conv, dim_conv, kernel_q, stride=stride_q,
                         padding=padding_q, groups=dim_conv, bias=False)
                if len(kernel_q) > 0 else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            
            self.pool_k = (
                nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv,
                         padding=padding_kv, groups=dim_conv, bias=False)
                if len(kernel_kv) > 0 else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            
            self.pool_v = (
                nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv,
                         padding=padding_kv, groups=dim_conv, bias=False)
                if len(kernel_kv) > 0 else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported mode {mode}")

    def _setup_relative_position_embeddings(self, patches_shape, stride_q, stride_kv, 
                                          head_dim, rel_pos_zero_init):
        """Setup relative positional embeddings."""
        assert len(patches_shape) == 2 and patches_shape[0] == patches_shape[1], \
            "Spatial relative position requires square input"
            
        size = patches_shape[0]
        q_size = size // stride_q[1] if len(stride_q) > 0 else size
        kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
        rel_sp_dim = 2 * max(q_size, kv_size) - 1

        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        if not rel_pos_zero_init:
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, hw_shape):
        B, N, _ = x.shape
        
        if self.pool_first:
            fold_dim = 1 if self.mode == "conv_unshared" else self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"
            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply pooling to Q, K, V
        q, q_shape = attention_pool(
            q, self.pool_q, hw_shape, 
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None)
        )
        k, k_shape = attention_pool(
            k, self.pool_k, hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None)
        )
        v, v_shape = attention_pool(
            v, self.pool_v, hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None)
        )

        if self.pool_first:
            # Apply linear projections after pooling
            q_N = np.prod(q_shape) + (1 if self.has_cls_embed else 0)
            k_N = np.prod(k_shape) + (1 if self.has_cls_embed else 0)
            v_N = np.prod(v_shape) + (1 if self.has_cls_embed else 0)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

        # Compute attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn, q, self.has_cls_embed, q_shape, k_shape,
                self.rel_pos_h, self.rel_pos_w
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        # Add residual connection if enabled
        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)

        return x


# class MultiScaleBlock(nn.Module):
#     """Multi-scale transformer block with pooled attention only (no MLP)."""
    
#     def __init__(
#         self,
#         dim_c,
#         dim_v,      
#         patches_shape,
#         hw_shape,
#         num_heads=8,
#         qkv_bias=False,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         kernel_q=(1, 1),
#         kernel_kv=(1, 1),
#         stride_q=(1, 1),
#         stride_kv=(1, 1),
#         mode="conv",
#         has_cls_embed=True,
#         pool_first=False,
#         rel_pos_spatial=False,
#         rel_pos_zero_init=False,
#         residual_pooling=True,
#         dim_mul_in_att=False,
        
#     ):
#         super().__init__()
#         self.hw_shape = hw_shape if hw_shape is not None else patches_shape
#         self.dim = dim_c
#         self.dim_out = dim_v
#         self.has_cls_embed = has_cls_embed
#         self.dim_mul_in_att = dim_mul_in_att
        
#         self.norm1 = norm_layer(dim_c)
        
#         att_dim = dim_v if dim_mul_in_att else dim_c
        
#         self.attn = MHPA(
#             dim_c, att_dim, patches_shape,
#             num_heads=num_heads, qkv_bias=qkv_bias,
#             kernel_q=kernel_q, kernel_kv=kernel_kv,
#             stride_q=stride_q, stride_kv=stride_kv,
#             norm_layer=norm_layer, has_cls_embed=has_cls_embed,
#             mode=mode, pool_first=pool_first,
#             rel_pos_spatial=rel_pos_spatial,
#             rel_pos_zero_init=rel_pos_zero_init,
#             residual_pooling=residual_pooling,
#         )
        
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#         # Dimension projection if needed
#         if dim_c != dim_v:
#             self.proj = nn.Linear(dim_c, dim_v)
#         else:
#             self.proj = None

#         # Skip connection pooling
#         if len(stride_q) > 0 and np.prod(stride_q) > 1:
#             kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
#             stride_skip = stride_q
#             padding_skip = [int(skip // 2) for skip in kernel_skip]
#             self.pool_skip = nn.MaxPool2d(
#                 kernel_skip, stride_skip, padding_skip, ceil_mode=False
#             )
#         else:
#             self.pool_skip = None

#     def forward(self, x):
#         hw_shape=self.hw_shape
#         x_norm = self.norm1(x)
#         x_block = self.attn(x_norm, hw_shape)

#         # Handle dimension change in attention
#         if self.dim_mul_in_att and self.proj is not None:
#             x = self.proj(x_norm)
#         elif not self.dim_mul_in_att and self.proj is not None:
#             x_norm = self.norm1(x)  # Re-normalize original input
#             x = self.proj(x_norm)
            
#         # Apply skip connection pooling
#         x_res, _ = attention_pool(
#             x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed
#         )
#         if x_res.shape != x_block.shape:
#             x_res = self.proj(x_res)  # Add projection layer

#         # Residual connection with attention output
#         x = x_res + self.drop_path(x_block)
        
#         return x
    
class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim_c,
        dim_v,      
        patches_shape,
        hw_shape,
        num_heads=8,
        qkv_bias=False,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",
        has_cls_embed=True,
        pool_first=False,
        rel_pos_spatial=False,
        rel_pos_zero_init=False,
        residual_pooling=True,
        dim_mul_in_att=False,
    ):
        super().__init__()
        self.hw_shape = hw_shape if hw_shape is not None else patches_shape
        self.dim = dim_c
        self.dim_out = dim_v
        self.has_cls_embed = has_cls_embed
        self.dim_mul_in_att = dim_mul_in_att
        
        self.norm1 = norm_layer(dim_c)
        
        att_dim = dim_v if dim_mul_in_att else dim_c
        
        self.attn = MHPA(
            dim_c, att_dim, patches_shape,
            num_heads=num_heads, qkv_bias=qkv_bias,
            kernel_q=kernel_q, kernel_kv=kernel_kv,
            stride_q=stride_q, stride_kv=stride_kv,
            norm_layer=norm_layer, has_cls_embed=has_cls_embed,
            mode=mode, pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Dimension projection
        if dim_c != dim_v:
            self.proj = nn.Linear(dim_c, dim_v)
        else:
            self.proj = None

        # Additional projection for pooled features if needed
        if dim_c != dim_v:
            self.pool_proj = nn.Linear(dim_c, dim_v)
        else:
            self.pool_proj = None


        # Skip connection pooling
        if np.prod(stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.pool_skip = nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
        else:
            self.pool_skip = None
        
    def forward(self, x):
        hw_shape = self.hw_shape
        x_norm = self.norm1(x)
        x_block = self.attn(x_norm, hw_shape)

        # Apply skip connection pooling
        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed
        )
        
        # Project residual if dimensions don't match
        if self.pool_proj is not None:
            x_res = self.pool_proj(x_res)
            
        # Residual connection with attention output
        x = x_res + self.drop_path(x_block)
        
        return x

class VarStreamDownSamplingBlock(nn.Module):
    """
    Downsamples the var stream using avg pool
    """

    def __init__(self, input_patches_shape, kernel_size, dim_in, dim_out):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()

        assert input_patches_shape[0] % kernel_size == 0
        assert input_patches_shape[1] % kernel_size == 0


        self.input_patches_shape = input_patches_shape  
        self.dim_in, self.dim_out = dim_in, dim_out
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=kernel_size)    

    def forward(self, X_1, X_2):

        B, _, _ = X_1.shape
        X_1 = X_1.transpose(1, 2).reshape(B, self.dim_in, self.input_patches_shape[0], self.input_patches_shape[1])
        X_1 = self.conv(X_1)
        X_1 = X_1.reshape(B, self.dim_out, -1).transpose(1, 2)

        return X_1, X_2
    
    
def main():
    """
    This is a simple test to check if the recomputation is correct
    by computing gradients of the first learnable parameters twice --
    once with the custom backward and once with the vanilla backward.

    The difference should be ~zero.
    """

    # insitantiating and fixing the model.
    # model = AsymmetricRevVit(
    #     const_dim=192,
    #     var_dim=[64, 128, 320, 512],
    #     sra_R=[8, 4, 2, 1],
    #     n_head=8,
    #     stages=[1, 1, 10, 1],
    #     drop_path_rate=0.1,
    #     patch_size=(
    #         4,
    #         4,
    #     ),  
    #     image_size=(224, 224),  
    #     num_classes=100,
    # )

    model = AsymmetricRevVit(
        const_dim=192,
        var_dim=[64, 128, 320, 512],
        sra_R=[8, 4, 2, 1],
        n_head=8,
        stages=[1, 1, 10, 1],
        drop_path_rate=0.1,
        patch_size=(
            4,
            4,
        ),  
        image_size=(224, 224),  
        num_classes=100,
    )

    # random input, instaintiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    x = torch.rand((1, 3, 224, 224))
    model = model
    
    # model = model.to("cuda")
    # x = x.to("cuda")
    import time
    start_time = time.time()          

    # output of the model under reversible backward logic
    output = model(x)
    # loss is just the norm of the output
    loss = output.norm(dim=1).mean()
    print(loss.shape)

    # computatin gradients with reversible backward logic
    # using retain_graph=True to keep the computation graph.
    loss.backward(retain_graph=True)

    end_time = time.time()

    print(f"Batch time: {(end_time - start_time) * 1000:.3f} ms") 

    # gradient of the patchification layer under custom bwd logic
    rev_grad = model.patch_embed1.weight.grad.clone()

    # resetting the computation graph
    for param in model.parameters():
        param.grad = None

    # switching model mode to use vanilla backward logic
    model.no_custom_backward = True

    # computing forward with the same input and model.
    output = model(x)
    # same loss
    loss = output.norm(dim=1)

    # backward but with vanilla logic, does not need retain_graph=True
    loss.backward()

    # looking at the gradient of the patchification layer again
    vanilla_grad = model.patch_embed1.weight.grad.clone()

    # difference between the two gradients is small enough.
    # assert (rev_grad - vanilla_grad).abs().max() < 1e-6

    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}\n")

    try:
        from fvcore.nn import FlopCountAnalysis
        model.training = False
        input = torch.randn(1, 3, 224, 224)
        flops = FlopCountAnalysis(model, input)
        # input = torch.randn(1, 3136, 192)
        # part = model.layers[4].F
        # flops = FlopCountAnalysis(part, input)
        # print(f"\nNumber of model parameters: {sum(p.numel() for p in part.parameters())}\n")
        print(f"Total MACs Estimate (fvcore): {flops.total()}")
    except:
        print("FLOPs estimator failed")
        pass

if __name__ == "__main__":
    main()
