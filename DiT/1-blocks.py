# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
block_outputs = torch.load('~/EOC/DiT/averaged_pth_att+mlp/averaged_block_outputs.pth')
    
def main(args):
    # Set up PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only the DiT-XL/2 model can be automatically downloaded."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Initialize the diffusion process
    diffusion = create_diffusion(str(args.num_sampling_steps))  

    # Load the model:
    latent_size = args.image_size // 8
    if args.accelerate_method is not None and args.accelerate_method == "dynamiclayer":
        from models.dynamic_models import DiT_models
    else:
        from models.models import DiT_models

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    if args.accelerate_method is not None and 'dynamiclayer' in args.accelerate_method:
        model.load_ranking(args.path, args.num_sampling_steps, diffusion.timestep_map, args.thres)
    
    # Automatically download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # Important!
    vae = AutoencoderKL.from_pretrained(f"~/EOC/DiT/pretrained_models/sd-vae-ft-ema").to(device)

    for i in range(1000):
        torch.manual_seed(args.seed)
        # Label for conditioning the model (can be freely changed):
        label = i
        class_labels = [label]

        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Set up classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # For collecting attention and MLP outputs from each block
        block_outputs = {}

        # Define a hook function to store attention and MLP outputs
        def hook_fn(module, input, reuse_feature, block_idx):
            if block_idx not in block_outputs:
                block_outputs[block_idx] = {'attn': [], 'mlp': []}

            # Extract attention and MLP outputs from the output
            attn_output, mlp_output = reuse_feature[1]  # Assume the forward returns (attn_output, mlp_output)
            # import pdb
            # pdb.set_trace()
            # print(attn_output.shape)
            # print(mlp_output.shape)
            block_outputs[block_idx]['attn'].append(attn_output)
            block_outputs[block_idx]['mlp'].append(mlp_output)

        # Register the hook to each block
        hooks = []
        for block_idx, block in enumerate(model.blocks):  # Assume the model has a blocks attribute
            hook = block.register_forward_hook(lambda module, input, reuse_feature, idx=block_idx: hook_fn(module, input, reuse_feature, idx))
            hooks.append(hook)
        
        _ = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, block_outputs=block_outputs
        )
        
        # Remove the hooks
        for hook in hooks:
            hook.remove()

        # Save block_outputs to the specified path
        save_path = f'~/learning-to-cache-main/DiT/50-block_pth_att+mlp/{label}_block_outputs_att+mlp.pth'
        torch.save(block_outputs, save_path)

        print("Attention and MLP outputs from each block have been saved to:", save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional DiT checkpoint path (default: automatically download the pre-trained DiT-XL/2 model).")
    parser.add_argument("--accelerate-method", type=str, default=None,
                        help="Use an accelerated version of the model.")
    
    parser.add_argument("--ddim-sample", action="store_true", default=False,)
    parser.add_argument("--p-sample", action="store_true", default=False,)
    
    parser.add_argument("--path", type=str, default=None,
                        help="Optional router checkpoint path")