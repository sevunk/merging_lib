import re
from typing import List, Optional

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from .utils import SafeTensorsWriter, aggressive_gc

def get_universal_alpha(key: str, preset_values: List[float], base_alpha: float) -> float:
    """Determine the alpha (blend weight) for a specific layer using elastic mapping.
    
    This function scales the user-provided weight preset to the detected 
    architecture's block structure (Flux, SDXL, Z-Image).
    
    Args:
        key: The tensor key (layer name).
        preset_values: List of weights to interpolate.
        base_alpha: Fallback weight if the layer isn't recognized as a block.
        
    Returns:
        The interpolated alpha value.
    """
    idx = -1
    max_blocks = 1

    # 1. FLUX (57 Blocks)
    if "double_blocks" in key or "single_blocks" in key:
        max_blocks = 57
        match_dbl = re.search(r"double_blocks\.(\d+)", key)
        if match_dbl:
            idx = int(match_dbl.group(1))
        else:
            match_sgl = re.search(r"single_blocks\.(\d+)", key)
            if match_sgl:
                idx = 19 + int(match_sgl.group(1))

    # 2. SDXL / PONY (20 Blocks)
    elif any(x in key for x in ["input_blocks", "middle_block", "output_blocks"]):
        max_blocks = 20
        match_in = re.search(r"input_blocks\.(\d+)", key)
        if match_in: 
            idx = int(match_in.group(1))
        elif "middle_block" in key: 
            idx = 9
        else:
            match_out = re.search(r"output_blocks\.(\d+)", key)
            if match_out: 
                idx = 10 + int(match_out.group(1))

    # 3. Z-IMAGE (30 Blocks)
    elif "layers." in key:
        max_blocks = 30
        match_lyr = re.search(r"layers\.(\d+)", key)
        if match_lyr: 
            idx = int(match_lyr.group(1))
    elif "embedder" in key or "pad_token" in key:
        idx = 0
        max_blocks = 30
    elif "refiner" in key or "final_layer" in key:
        idx = 29
        max_blocks = 30

    if idx != -1:
        # Interpolate the user preset across the detected block range
        user_x = np.linspace(0, max_blocks - 1, len(preset_values))
        alpha = np.interp(idx, user_x, preset_values)
        return float(alpha)

    return base_alpha

def mbw_merge(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    preset_weights: List[float],
    base_alpha: float = 0.5,
    method: str = "Weighted Sum",
    precision: str = "fp16"
) -> None:
    """Perform MBW (Merge by Weight) streaming merge.
    
    Args:
        model_a_path: Path to Model A (recipient).
        model_b_path: Path to Model B (donor).
        output_path: Path to save the merged model.
        preset_weights: List of weights for block interpolation.
        base_alpha: Default alpha for layers outside recognized blocks.
        method: Merging method ('Weighted Sum', 'SLERP').
        precision: Output precision ('fp16', 'fp8', 'fp32').
    """
    print(f"🌍 Starting MBW Merge ({method})")
    writer = SafeTensorsWriter(output_path, precision)
    
    from .utils import slerp_tensor

    with safe_open(model_a_path, framework="pt", device="cpu") as f_a, \
         safe_open(model_b_path, framework="pt", device="cpu") as f_b:
        
        all_keys = sorted(list(f_a.keys()))
        keys_b = set(f_b.keys())
        
        # Build Index
        for key in all_keys:
            writer.add_to_index(key, f_a.get_slice(key).get_shape())
            
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            
            for key in tqdm(all_keys, desc="MBW Progress"):
                try:
                    t_a = f_a.get_tensor(key)
                    
                    if key not in keys_b:
                        result = t_a
                    else:
                        t_b = f_b.get_tensor(key)
                        if t_a.shape != t_b.shape:
                            result = t_a
                        else:
                            alpha = get_universal_alpha(key, preset_weights, base_alpha)
                            
                            if method == "SLERP":
                                result = slerp_tensor(t_a, t_b, alpha)
                            else:
                                result = (t_a.float() * (1.0 - alpha) + t_b.float() * alpha).to(t_a.dtype)
                        del t_b
                        
                    writer.write_tensor(f_out, result)
                    del t_a, result
                    
                except Exception as e:
                    print(f"\n⚠️ Error on layer {key}: {e}")
                    writer.write_tensor(f_out, f_a.get_tensor(key))
                
                aggressive_gc()

    print(f"✅ MBW Merge complete: {output_path}")

def nuclear_merge(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    alpha: float = 0.5,
    method: str = "Weighted Sum",
    te_source: str = "Weighted Sum",
    precision: str = "fp16"
) -> None:
    """Universal/Nuclear Merge: A streamlined 2-model merge with selective Text Encoder choice.
    
    Args:
        model_a_path: Path to Model A.
        model_b_path: Path to Model B.
        output_path: Path to save result.
        alpha: Blend weight (0.0 = all A, 1.0 = all B).
        method: 'Weighted Sum' or 'SLERP'.
        te_source: 'Weighted Sum', 'Copy Model A', or 'Copy Model B'.
        precision: Output precision.
    """
    print(f"☢️ Starting NUCLEAR MERGE ({method})")
    print(f"   TE Source: {te_source}")
    writer = SafeTensorsWriter(output_path, precision)
    
    from .utils import slerp_tensor

    with safe_open(model_a_path, framework="pt", device="cpu") as f_a, \
         safe_open(model_b_path, framework="pt", device="cpu") as f_b:
        
        keys_a = set(f_a.keys())
        keys_b = set(f_b.keys())
        all_keys = sorted(list(keys_a | keys_b))
        
        for k in all_keys:
            if k in keys_a: writer.add_to_index(k, f_a.get_slice(k).get_shape())
            else: writer.add_to_index(k, f_b.get_slice(k).get_shape())
            
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            
            for key in tqdm(all_keys, desc="Nuclear Processing"):
                try:
                    is_te = any(x in key for x in ["text_model", "te_", "text_encoder", "conditioner", "t5xxl"])
                    
                    # 1. Handle Selective Text Encoder
                    if is_te:
                        if te_source == "Copy Model A" and key in keys_a:
                            writer.write_tensor(f_out, f_a.get_tensor(key))
                            continue
                        elif te_source == "Copy Model B" and key in keys_b:
                            writer.write_tensor(f_out, f_b.get_tensor(key))
                            continue
                    
                    # 2. Standard Merge
                    if key in keys_a and key in keys_b:
                        t_a = f_a.get_tensor(key)
                        t_b = f_b.get_tensor(key)
                        
                        if t_a.shape != t_b.shape:
                            result = t_a # Fallback
                        else:
                            if method == "SLERP":
                                result = slerp_tensor(t_a, t_b, alpha)
                            else:
                                result = (t_a.float() * (1.0 - alpha) + t_b.float() * alpha).to(t_a.dtype)
                        del t_a, t_b
                    elif key in keys_a:
                        result = f_a.get_tensor(key)
                    else:
                        result = f_b.get_tensor(key)
                        
                    writer.write_tensor(f_out, result)
                    del result
                    
                except Exception as e:
                    print(f"\n⚠️ Skip {key}: {e}")
                
                aggressive_gc()

    print(f"✅ Nuclear Merge complete: {output_path}")
