import torch
from torch import nn
from torch.nn import MultiheadAttention as MHA
from typing import Tuple, Optional, List
import math

# Import token mixers - you'll need to ensure these are also TorchScript compatible
try:
    from tokenmixers import PoolingFBlock, SpatialMLPFBlock
except ImportError:
    # Fallback implementations if tokenmixers not available
    class PoolingFBlock(nn.Module):
        def __init__(self, dim: int, pool_size: int, patches_shape: Tuple[int, int], 
                     num_registers: int, enable_amp: bool = False):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.enable_amp = enable_amp
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                return self.norm(x)
    
    class SpatialMLPFBlock(nn.Module):
        def __init__(self, dim: int, patches_shape: Tuple[int, int], enable_amp: bool = False):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.enable_amp = enable_amp
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                return self.norm(x)

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = True) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class MLPSubblock(nn.Module):
    """
    MLP sub-block with pre-LayerNorm.
    """
    def __init__(self, dim: int, mlp_ratio: int = 4, enable_amp: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.enable_amp = enable_amp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            return self.mlp(self.norm(x))

class AttentionSubBlock(nn.Module):
    """
    Attention sub-block with pre-LayerNorm.
    """
    def __init__(self, dim: int, num_heads: int, enable_amp: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MHA(dim, num_heads, batch_first=True)
        self.enable_amp = enable_amp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            x = self.norm(x)
            out, _ = self.attn(x, x, x)
            return out

class ReversibleBlock(nn.Module):
    """
    TorchScript compatible Reversible Block.
    Note: This version uses standard backpropagation instead of custom reversible backprop.
    """
    def __init__(self, dim: int, num_heads: int, enable_amp: bool, drop_path: float, 
                 token_mixer: str, pool_size: int, patches_shape: Tuple[int, int], 
                 num_registers: int):
        super().__init__()
        self.drop_path_rate = drop_path
        self.token_mixer = token_mixer
        
        # Initialize F block - use explicit if-else for TorchScript
        if token_mixer == "attention":
            self.F = AttentionSubBlock(dim=dim, num_heads=num_heads, enable_amp=enable_amp)
        elif token_mixer == "pooling":
            self.F = PoolingFBlock(
                dim=dim, pool_size=pool_size, patches_shape=patches_shape, 
                num_registers=num_registers, enable_amp=enable_amp
            )
        elif token_mixer == "spatial_mlp":
            self.F = SpatialMLPFBlock(dim=dim, patches_shape=patches_shape, enable_amp=enable_amp)
        else:
            # Default case - TorchScript needs all paths to be defined
            self.F = AttentionSubBlock(dim=dim, num_heads=num_heads, enable_amp=enable_amp)
        
        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

    def forward(self, X_1: torch.Tensor, X_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass (not memory-efficient reversible version).
        Y_1 = X_1 + F(X_2)
        Y_2 = X_2 + G(Y_1)
        """
        # F block
        f_X_2 = self.F(X_2)
        f_X_2_dropped = drop_path(f_X_2, drop_prob=self.drop_path_rate, training=self.training)
        Y_1 = X_1 + f_X_2_dropped
        
        # G block
        g_Y_1 = self.G(Y_1)
        g_Y_1_dropped = drop_path(g_Y_1, drop_prob=self.drop_path_rate, training=self.training)
        Y_2 = X_2 + g_Y_1_dropped
        
        return Y_1, Y_2

class RevViT(nn.Module):
    """
    TorchScript compatible Reversible Vision Transformer.
    Note: This version uses standard backpropagation for TorchScript compatibility.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        n_head: int = 8,
        depth: int = 8,
        drop_path_rate: float = 0.0,
        patch_size: Tuple[int, int] = (2, 2),
        image_size: Tuple[int, int] = (32, 32),
        num_classes: int = 10,
        enable_amp: bool = False,
        token_mixer: str = "attention",
        pool_size: int = 3,
        num_registers: int = 0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size
        self.num_registers = num_registers
        
        # Calculate number of patches
        num_patches = (image_size[0] // self.patch_size[0]) * (image_size[1] // self.patch_size[1])
        patches_shape = (image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1])
        
        # Always create reg_tokens parameter, but initialize to zeros if not used
        self.reg_tokens = nn.Parameter(torch.zeros(1, max(1, self.num_registers), embed_dim))
        
        # Stochastic depth decay rule - make this deterministic for TorchScript
        drop_rates = []
        for i in range(depth):
            drop_rates.append(drop_path_rate * i / max(1, depth - 1))
        
        # Create layers list explicitly to avoid list comprehension issues
        layers = []
        for i in range(depth):
            layer = ReversibleBlock(
                dim=embed_dim,
                num_heads=n_head,
                enable_amp=enable_amp,
                drop_path=drop_rates[i],
                token_mixer=token_mixer,
                pool_size=pool_size,
                patches_shape=patches_shape,
                num_registers=num_registers
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        
        # Standard components
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, num_patches + num_registers, embed_dim))
        
        # Head for final projection
        self.head = nn.Linear(2 * embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(2 * embed_dim)
        
        # For compatibility with original interface
        self.no_custom_backward: bool = True  # Always use standard backprop in TorchScript

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patchification
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add register tokens if present
        if self.num_registers > 0:
            batch_registers = self.reg_tokens[:, :self.num_registers, :].expand(x.shape[0], -1, -1)
            x = torch.cat([batch_registers, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embeddings
        
        # Initialize two streams
        X_1 = x
        X_2 = x
        
        # Pass through reversible blocks
        for layer in self.layers:
            X_1, X_2 = layer(X_1, X_2)
        
        # Concatenate streams
        x = torch.cat([X_1, X_2], dim=-1)
        
        # Global average pooling
        x = x.mean(1)
        
        # Head pre-norm and classification
        x = self.norm(x)
        x = self.head(x)
        
        return x

# Factory function for easier model creation
def create_revvit(
    embed_dim: int = 768,
    n_head: int = 8,
    depth: int = 8,
    drop_path_rate: float = 0.0,
    patch_size: Tuple[int, int] = (16, 16),
    image_size: Tuple[int, int] = (224, 224),
    num_classes: int = 1000,
    enable_amp: bool = False,
    token_mixer: str = "attention",
    pool_size: int = 3,
    num_registers: int = 0
) -> RevViT:
    """Create a TorchScript compatible RevViT model."""
    return RevViT(
        embed_dim=embed_dim,
        n_head=n_head,
        depth=depth,
        drop_path_rate=drop_path_rate,
        patch_size=patch_size,
        image_size=image_size,
        num_classes=num_classes,
        enable_amp=enable_amp,
        token_mixer=token_mixer,
        pool_size=pool_size,
        num_registers=num_registers
    )

def create_scriptable_revvit(**kwargs) -> torch.jit.ScriptModule:
    """Create a TorchScript compiled RevViT model."""
    model = create_revvit(**kwargs)
    model.eval()
    
    # Create example input for tracing
    image_size = kwargs.get('image_size', (224, 224))
    example_input = torch.randn(1, 3, image_size[0], image_size[1])
    
    # Compile to TorchScript using tracing
    scripted_model = torch.jit.trace(model, example_input)
    return scripted_model

# Example usage and testing
def test_torchscript_compatibility():
    """Test TorchScript compatibility of the RevViT model."""
    print("Testing TorchScript compatibility...")
    
    # Create model with simple configuration
    model = create_revvit(
        embed_dim=192,
        n_head=3,
        depth=6,
        image_size=(224, 224),
        patch_size=(16, 16),
        num_classes=1000,
        num_registers=0,  # Start without registers
        token_mixer="attention"  # Use attention mixer
    )
    
    model.eval()
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)  # Use batch size 1 for tracing
    
    try:
        output = model(x)
        print(f"Model output shape: {output.shape}")
        
        # Test TorchScript tracing (more reliable than scripting)
        traced_model = torch.jit.trace(model, x)
        traced_output = traced_model(x)
        print(f"TorchScript traced output shape: {traced_output.shape}")
        
        # Verify outputs match
        diff = torch.abs(output - traced_output).max()
        print(f"Max difference between original and traced: {diff.item()}")
        
        if diff.item() < 1e-5:
            print("✓ TorchScript tracing successful!")
            return True, traced_model
        else:
            print("✗ Outputs don't match closely enough")
            return False, None
            
    except Exception as e:
        print(f"✗ TorchScript compilation failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False, None

if __name__ == "__main__":
    success, scripted_model = test_torchscript_compatibility()
    if success:
        print("TorchScript compatibility test passed!")
    else:
        print("TorchScript compatibility test failed!")
    scripted_model.save("scripted_revvit.pt")

