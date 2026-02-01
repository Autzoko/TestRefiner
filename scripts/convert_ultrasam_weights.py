#!/usr/bin/env python
"""Convert UltraSAM mmdet checkpoint to standalone PyTorch weights.

Usage:
    python scripts/convert_ultrasam_weights.py \
        --input weights/UltraSam.pth \
        --output weights/ultrasam_standalone.pth
"""

import argparse
import re
import torch
from collections import OrderedDict


def build_key_mapping():
    """Define the key mapping rules from mmdet to standalone."""
    rules = [
        # Image encoder (backbone)
        (r'^backbone\.patch_embed\.projection\.(.*)', r'image_encoder.patch_embed.proj.\1'),
        (r'^backbone\.pos_embed', r'image_encoder.pos_embed'),
        (r'^backbone\.layers\.(\d+)\.ln1\.(.*)', r'image_encoder.blocks.\1.norm1.\2'),
        (r'^backbone\.layers\.(\d+)\.ln2\.(.*)', r'image_encoder.blocks.\1.norm2.\2'),
        (r'^backbone\.layers\.(\d+)\.attn\.qkv\.(.*)', r'image_encoder.blocks.\1.attn.qkv.\2'),
        (r'^backbone\.layers\.(\d+)\.attn\.proj\.(.*)', r'image_encoder.blocks.\1.attn.proj.\2'),
        (r'^backbone\.layers\.(\d+)\.attn\.rel_pos_h', r'image_encoder.blocks.\1.attn.rel_pos_h'),
        (r'^backbone\.layers\.(\d+)\.attn\.rel_pos_w', r'image_encoder.blocks.\1.attn.rel_pos_w'),
        (r'^backbone\.layers\.(\d+)\.ffn\.layers\.0\.0\.(.*)', r'image_encoder.blocks.\1.mlp.lin1.\2'),
        (r'^backbone\.layers\.(\d+)\.ffn\.layers\.1\.(.*)', r'image_encoder.blocks.\1.mlp.lin2.\2'),
        # Neck
        (r'^backbone\.neck\.0\.(.*)', r'image_encoder.neck.0.\1'),
        (r'^backbone\.neck\.1\.(.*)', r'image_encoder.neck.1.\1'),
        (r'^backbone\.neck\.2\.(.*)', r'image_encoder.neck.2.\1'),
        (r'^backbone\.neck\.3\.(.*)', r'image_encoder.neck.3.\1'),

        # Prompt encoder
        (r'^prompt_encoder\.(.*)', r'prompt_encoder.\1'),

        # Transformer decoder
        (r'^decoder\.layers\.(\d+)\.self_attn\.(.*)', r'transformer_decoder.layers.\1.self_attn.\2'),
        (r'^decoder\.layers\.(\d+)\.cross_attn_token_to_image\.(.*)', r'transformer_decoder.layers.\1.cross_attn_token_to_image.\2'),
        (r'^decoder\.layers\.(\d+)\.cross_attn_image_to_token\.(.*)', r'transformer_decoder.layers.\1.cross_attn_image_to_token.\2'),
        (r'^decoder\.layers\.(\d+)\.mlp\.(.*)', r'transformer_decoder.layers.\1.mlp.\2'),
        (r'^decoder\.layers\.(\d+)\.norm1\.(.*)', r'transformer_decoder.layers.\1.norm1.\2'),
        (r'^decoder\.layers\.(\d+)\.norm2\.(.*)', r'transformer_decoder.layers.\1.norm2.\2'),
        (r'^decoder\.layers\.(\d+)\.norm3\.(.*)', r'transformer_decoder.layers.\1.norm3.\2'),
        (r'^decoder\.layers\.(\d+)\.norm4\.(.*)', r'transformer_decoder.layers.\1.norm4.\2'),
        (r'^decoder\.final_attn_token_to_image\.(.*)', r'transformer_decoder.final_attn_token_to_image.\1'),
        (r'^decoder\.post_norm\.(.*)', r'transformer_decoder.post_norm.\1'),

        # Mask decoder (bbox_head)
        (r'^bbox_head\.output_upscaling\.(.*)', r'mask_decoder.output_upscaling.\1'),
        (r'^bbox_head\.output_hypernetworks_mlps\.(.*)', r'mask_decoder.output_hypernetworks_mlps.\1'),
        (r'^bbox_head\.iou_prediction_head\.(.*)', r'mask_decoder.iou_prediction_head.\1'),
    ]
    return rules


def map_key(key, rules):
    """Map a single key using the rule list. Returns new key or None if unmapped."""
    for pattern, replacement in rules:
        new_key, n = re.subn(pattern, replacement, key)
        if n > 0:
            return new_key
    return None


def convert_checkpoint(input_path, output_path):
    """Load mmdet checkpoint and convert to standalone format."""
    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)

    if 'state_dict' in ckpt:
        src_state = ckpt['state_dict']
        print(f"  Found 'state_dict' key with {len(src_state)} parameters")
    elif 'model' in ckpt:
        src_state = ckpt['model']
        print(f"  Found 'model' key with {len(src_state)} parameters")
    else:
        src_state = ckpt
        print(f"  Using checkpoint directly with {len(src_state)} parameters")

    rules = build_key_mapping()
    new_state = OrderedDict()
    unmapped = []
    mapped_count = 0

    # First pass: direct mapping
    for key, value in src_state.items():
        new_key = map_key(key, rules)
        if new_key is not None:
            new_state[new_key] = value
            mapped_count += 1
        else:
            unmapped.append(key)

    # Second pass: strip 'sam.' prefix for unmapped keys
    still_unmapped = []
    for key in unmapped:
        if key.startswith('sam.'):
            stripped = key[4:]
            new_key = map_key(stripped, rules)
            if new_key is not None:
                new_state[new_key] = src_state[key]
                mapped_count += 1
                continue
        still_unmapped.append(key)

    print(f"\nConversion summary:")
    print(f"  Mapped:   {mapped_count} / {len(src_state)} parameters")
    print(f"  Unmapped: {len(still_unmapped)} parameters")

    if still_unmapped:
        print(f"\nUnmapped keys ({len(still_unmapped)}):")
        for k in sorted(still_unmapped):
            print(f"  {k}")

    module_counts = {}
    for key in new_state:
        module = key.split('.')[0]
        module_counts[module] = module_counts.get(module, 0) + 1
    print(f"\nParameters per module:")
    for mod, cnt in sorted(module_counts.items()):
        print(f"  {mod}: {cnt}")

    torch.save(new_state, output_path)
    print(f"\nSaved standalone weights to: {output_path}")
    print(f"Total parameters: {sum(p.numel() for p in new_state.values()):,}")

    return new_state, still_unmapped


def verify_against_model(state_dict):
    """Verify converted weights can be loaded into the standalone model."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.ultrasam import build_ultrasam_vit_b

    model = build_ultrasam_vit_b()
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"\nMissing keys in checkpoint ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}")
    if unexpected:
        print(f"\nUnexpected keys in checkpoint ({len(unexpected)}):")
        for k in sorted(unexpected):
            print(f"  {k}")
    if not missing and not unexpected:
        print("\nAll keys match perfectly!")

    result = model.load_state_dict(state_dict, strict=False)
    print(f"\nload_state_dict result:")
    print(f"  Missing keys:    {len(result.missing_keys)}")
    print(f"  Unexpected keys: {len(result.unexpected_keys)}")

    return len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert UltraSAM mmdet weights to standalone')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to mmdet UltraSAM checkpoint (e.g., UltraSam.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for standalone weights')
    parser.add_argument('--verify', action='store_true',
                        help='Verify converted weights against model architecture')
    parser.add_argument('--print-src-keys', action='store_true',
                        help='Print all source checkpoint keys')
    parser.add_argument('--print-dst-keys', action='store_true',
                        help='Print all converted keys')
    args = parser.parse_args()

    state_dict, unmapped = convert_checkpoint(args.input, args.output)

    if args.print_src_keys:
        ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
        print("\n=== Source keys ===")
        for k in sorted(sd.keys()):
            print(f"  {k}  {tuple(sd[k].shape)}")

    if args.print_dst_keys:
        print("\n=== Converted keys ===")
        for k in sorted(state_dict.keys()):
            print(f"  {k}  {tuple(state_dict[k].shape)}")

    if args.verify:
        verify_against_model(state_dict)
