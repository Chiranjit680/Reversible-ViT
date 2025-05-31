from rev import *
from fast_rev import *
from vit import *
# from revseg import HierarchicalRevVit, OnlyMLPRevViT
from asymmrev import AsymmetricRevVit

import timm
import torchprofile

def build_model(args):
    if args.model == "vit":
        if args.pareprop:
            rev_arch = FastRevViT
        else:
            rev_arch = RevViT

        model = rev_arch(
            embed_dim=args.embed_dim,
            n_head=args.n_head,
            depth=args.depth,
            drop_path_rate=(0.1 if args.deit_scheme else 0.0),
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
            token_mixer=args.token_mixer,
            pool_size=args.pool_size,
            num_registers=args.num_registers
        )
    # elif args.model == "swin":
    #     model = RevSwin(
    #         img_size=args.image_size,
    #         patch_size=args.patch_size,
    #         num_classes=args.num_classes,
    #         embed_dim=args.embed_dim,
    #         depths=[args.depth // 2, args.depth // 2],
    #         num_heads=[args.n_head, args.n_head * 2],
    #         window_size=4,
    #         fast_backprop=args.pareprop,
    #     )
    # elif args.model == "mvit":
    #     model = RevMViT(
    #         img_size=args.image_size,
    #         patch_kernel=(3, 3),
    #         patch_stride=(2, 2),
    #         patch_padding=(1, 1),
    #         num_classes=args.num_classes,
    #         embed_dim=args.embed_dim,
    #         depth=args.depth,
    #         num_heads=args.n_head,  # doubles every stage
    #         last_block_indexes=[0, 2],
    #         qkv_pool_kernel=(3, 3),
    #         adaptive_kv_stride=2,
    #         adaptive_window_size=16,
    #         fast_backprop=args.pareprop,
    #     )
    elif args.model == "vit-og":
        model = ViT_OG(
            embed_dim=args.embed_dim,
            n_head=args.n_head,
            depth=args.depth,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
            token_mixer=args.token_mixer,
            pool_size=args.pool_size,
            num_registers=args.num_registers
        )
    elif args.model == "vit-small":
        print("Warning : vit-small is not configured to its native 224x224 image and 16x16 patch size")
        print("Warning : vit-small modified to not use class tokens and classifier head is uses average pool of output sequence")
        if args.num_registers > 0:
            print("Registers with timm vit-small? Are you sure?")
            quit()
        model = timm.create_model("vit_small_patch16_224", pretrained=False, 
                                    num_classes=args.num_classes, img_size=args.image_size, patch_size=args.patch_size,
                                    class_token=False, global_pool='avg', drop_rate=0.0, drop_path_rate=(0.1 if args.deit_scheme else 0.0))
    elif args.model == "vit-hr-small":
        model = HierarchicalRevVit(
            embed_dim=args.embed_dim,
            n_head=args.n_head,
            stages=[3, 3, 6, 3],
            drop_path_rate=(0.1 if args.deit_scheme else 0.0),
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
            token_mixer=args.token_mixer,
            pool_size=args.pool_size
        )
    elif args.model == "onlymlp-revvit":
        model = OnlyMLPRevViT(
            embed_dim=args.embed_dim,
            stages=[3, 3, 6, 3],
            drop_path_rate=(0.1 if args.deit_scheme else 0.0),
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
        )
    elif args.model == "asymm-revvit":
        model = AsymmetricRevVit(
            const_dim=args.embed_dim,
            var_dim=[64, 128, 320, 512],
            sra_R=[8, 4, 2, 1],
            n_head=args.n_head,
            stages=[1, 1, 10, 1],
            drop_path_rate=(0.1 if args.deit_scheme else 0.0),
            patch_size=args.patch_size,  
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
        )
    else:
        raise NotImplementedError(f"Model {args.model} not supported.")
    
    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}\n")

    try:
        from fvcore.nn import FlopCountAnalysis
        model.training = False
        input = torch.randn(1, 3, 224, 224)
        flops = FlopCountAnalysis(model, input)
        print(f"Total MACs Estimate (fvcore): {flops.total()}")
    except:
        print("FLOPs estimator failed")
        pass

    # Whether to use memory-efficient reversible backpropagation or vanilla backpropagation
    # Note that in both cases, the model is reversible.
    # For Swin, this requires iterating through the layers.
    model.no_custom_backward = args.vanilla_bp
    
    return model
