import os
import json
import struct
import re
from typing import Dict, List, Set, Optional, Tuple, Union

import torch
from safetensors import safe_open
from tqdm import tqdm

from .utils import SafeTensorsWriter, aggressive_gc, clean_tensor, smart_convert_lora_key

def is_target_block(key: str, target_mode: str) -> bool:
    """Check if a layer key belongs to a specific architectural block for swapping.
    
    Args:
        key: The tensor key.
        target_mode: The swap target (e.g., 'SDXL_Input', 'FLUX_Double').
        
    Returns:
        True if the key is in the target block.
    """
    if target_mode == "SDXL_Input" and "input_blocks" in key: return True
    if target_mode == "SDXL_Middle" and "middle_block" in key: return True
    if target_mode == "SDXL_Output" and "output_blocks" in key: return True
    if target_mode == "FLUX_Double" and "double_blocks" in key: return True
    if target_mode == "FLUX_Single" and "single_blocks" in key: return True
    if target_mode == "Z_Embedders" and ("embedder" in key or "pad_token" in key): return True
    if target_mode == "Z_Layers" and "layers." in key: return True
    if target_mode == "Z_Refiners" and ("refiner" in key or "final_layer" in key): return True
    return False

def block_swap(
    body_model_path: str,
    donor_model_path: str,
    output_path: str,
    swap_target: str,
    precision: str = "fp16"
) -> None:
    """Perform 'Frankenstein' block transplantation between two models.
    
    Args:
        body_model_path: The recipient model path.
        donor_model_path: The donor model path.
        output_path: Path to save the hybrid model.
        swap_target: Architecture-specific block to transplant.
        precision: Output precision ('fp16', 'fp8', 'fp32').
    """
    print(f"🧬 Starting Block Swap (Target: {swap_target})")
    writer = SafeTensorsWriter(output_path, precision)
    
    with safe_open(body_model_path, framework="pt", device="cpu") as f_body, \
         safe_open(donor_model_path, framework="pt", device="cpu") as f_donor:
        
        body_keys = sorted(list(f_body.keys()))
        donor_keys = set(f_donor.keys())
        
        for k in body_keys:
            if is_target_block(k, swap_target) and k in donor_keys:
                writer.add_to_index(k, f_donor.get_slice(k).get_shape())
            else:
                writer.add_to_index(k, f_body.get_slice(k).get_shape())
                
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            swapped_count = 0
            
            for key in tqdm(body_keys, desc="Transplanting Blocks"):
                if is_target_block(key, swap_target) and key in donor_keys:
                    result = f_donor.get_tensor(key)
                    swapped_count += 1
                else:
                    result = f_body.get_tensor(key)
                    
                writer.write_tensor(f_out, result)
                del result
                aggressive_gc()

    print(f"✅ Swap complete! {swapped_count} layers transplanted. Output: {output_path}")

def svd_merge_loras(
    lora_a_path: str,
    lora_b_path: str,
    output_path: str,
    target_rank: int = 64,
    ratio_a: float = 1.0,
    ratio_b: float = 1.0,
    precision: str = "fp16"
) -> None:
    """Merge two LoRAs using SVD compression to a fixed rank.
    
    Supports both Linear and Convolutional layers, and matches mixed key formats.
    """
    print(f"🗜️ Starting SVD LoRA Merge (Rank: {target_rank})")
    writer = SafeTensorsWriter(output_path, precision)
    
    with safe_open(lora_a_path, framework="pt", device="cpu") as f_a, \
         safe_open(lora_b_path, framework="pt", device="cpu") as f_b:
        
        def get_bases_map(keys):
            m = {}
            for k in keys:
                base = smart_convert_lora_key(k)
                if base not in m: m[base] = {}
                if "lora_up" in k: m[base]["up"] = k
                elif "lora_down" in k: m[base]["down"] = k
                elif "alpha" in k: m[base]["alpha"] = k
            return m
            
        map_a = get_bases_map(f_a.keys())
        map_b = get_bases_map(f_b.keys())
        common_bases = sorted(list(set(map_a.keys()).intersection(set(map_b.keys()))))
        
        tensors_out = {}
        
        for base in tqdm(common_bases, desc="SVD Processing"):
            try:
                def get_w(f, m_base):
                    up = f.get_tensor(m_base["up"]).float()
                    down = f.get_tensor(m_base["down"]).float()
                    alpha = f.get_tensor(m_base["alpha"]).item() if "alpha" in m_base else down.shape[0]
                    
                    # Store original shapes for reconstruction
                    orig_shape = (up.shape[0], down.shape[1], up.shape[2], up.shape[3]) if len(up.shape) == 4 else (up.shape[0], down.shape[1])
                    
                    if len(up.shape) == 4: # Conv
                        up_2d = up.view(up.shape[0], -1)
                        down_2d = down.view(down.shape[0], -1)
                        w_2d = up_2d @ down_2d
                        return w_2d, orig_shape, (up.shape, down.shape), alpha
                    else:
                        return up @ down, orig_shape, (up.shape, down.shape), alpha
                
                # Check if Up/Down pairs exist for this base in both
                if "up" not in map_a[base] or "down" not in map_a[base]: continue
                if "up" not in map_b[base] or "down" not in map_b[base]: continue

                w_a, shape, raw_shapes, alpha_a = get_w(f_a, map_a[base])
                w_b, _, _, alpha_b = get_w(f_b, map_b[base])
                
                # Apply scaling and ratios
                full_delta = (w_a * (alpha_a / raw_shapes[0][1]) * ratio_a) + \
                             (w_b * (alpha_b / raw_shapes[1][1]) * ratio_b)
                
                # Singular Value Decomposition
                U, S, Vh = torch.linalg.svd(full_delta, full_matrices=False)
                U = U[:, :target_rank]
                S = S[:target_rank]
                Vh = Vh[:target_rank, :]
                
                sqrt_s = torch.diag(torch.sqrt(S))
                new_up_2d = U @ sqrt_s
                new_down_2d = sqrt_s @ Vh
                
                # Reshape back to original 4D for Conv
                if len(shape) == 4:
                    new_up = new_up_2d.view(shape[0], target_rank, 1, 1)
                    new_down = new_down_2d.view(target_rank, shape[1], shape[2], shape[3])
                else:
                    new_up = new_up_2d
                    new_down = new_down_2d
                
                tensors_out[base + ".lora_up.weight"] = new_up
                tensors_out[base + ".lora_down.weight"] = new_down
                tensors_out[base + ".alpha"] = torch.tensor([float(target_rank)])
                
                del w_a, w_b, full_delta, U, S, Vh, new_up, new_down
            except Exception as e:
                pass # Silent fail for incompatible layers

        for k, v in tensors_out.items():
            writer.add_to_index(k, v.shape)
            
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            for k in tqdm(tensors_out.keys(), desc="Writing SVD result"):
                writer.write_tensor(f_out, tensors_out[k])
                
    print(f"✅ SVD Merge complete: {output_path}")

def model_modify(
    model_path: str,
    output_path: str,
    target_block: str = "all",
    specific_ids: Optional[List[int]] = None,
    scale: float = 1.0,
    precision: str = "fp16"
) -> None:
    """Modify weights of specific blocks in a model by scaling them."""
    print(f"🛠️ Modifying Model (Target: {target_block}, Scale: {scale})")
    writer = SafeTensorsWriter(output_path, precision)
    
    def should_edit(key: str) -> bool:
        if target_block == "all": return True
        if target_block not in key: return False
        if not specific_ids: return True
        
        match = re.search(rf"{target_block}\.(\d+)\.", key)
        if match:
            idx = int(match.group(1))
            return idx in specific_ids
        return False

    with safe_open(model_path, framework="pt", device="cpu") as f:
        all_keys = sorted(list(f.keys()))
        for k in all_keys:
            writer.add_to_index(k, f.get_slice(k).get_shape())
            
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            for key in tqdm(all_keys, desc="Modifying"):
                tensor = f.get_tensor(key)
                if should_edit(key) and key.endswith(".weight"):
                    tensor = tensor.float() * scale
                writer.write_tensor(f_out, tensor)
                del tensor
                aggressive_gc()

    print(f"✅ Modification complete. Saved to: {output_path}")

def vae_fixer(
    input_model_path: str,
    output_model_path: str,
    contrast: float = 1.05,
    brightness: float = 0.0,
    target_keywords: List[str] = ["decoder", "post_quant_conv"],
    precision: str = "fp16"
) -> None:
    """Calibrate VAE Decoder layers for contrast and brightness."""
    print(f"🎨 Fixing VAE (Contrast: {contrast}, Brightness: {brightness})")
    writer = SafeTensorsWriter(output_model_path, precision)
    
    with safe_open(input_model_path, framework="pt", device="cpu") as f_in:
        all_keys = sorted(list(f_in.keys()))
        for k in all_keys:
            writer.add_to_index(k, f_in.get_slice(k).get_shape())
            
        with open(output_model_path, "wb") as f_out:
            writer.write_header(f_out)
            for key in tqdm(all_keys, desc="Fixing VAE"):
                tensor = f_in.get_tensor(key)
                is_target = any(kw in key for kw in target_keywords)
                is_vae = any(v in key for v in ["first_stage_model", "vae", "decoder"])
                
                if is_target and is_vae:
                    tensor = tensor.float()
                    if "weight" in key:
                        tensor = tensor * contrast
                    if "bias" in key:
                        tensor = tensor + brightness
                
                writer.write_tensor(f_out, tensor)
                del tensor
                aggressive_gc()

    print(f"✅ VAE Fix complete. Saved to: {output_model_path}")

def extract_lora(
    tuned_model_path: str,
    base_model_path: str,
    output_lora_path: str,
    threshold: float = 1e-4,
    precision: str = "fp16"
) -> None:
    """Extract full-rank diff (LoRA) between two models."""
    print(f"⛏️ Extracting LoRA (Threshold: {threshold})")
    writer = SafeTensorsWriter(output_lora_path, precision)
    
    with safe_open(tuned_model_path, framework="pt", device="cpu") as f_tuned, \
         safe_open(base_model_path, framework="pt", device="cpu") as f_base:
        
        common_keys = sorted(list(set(f_tuned.keys()).intersection(set(f_base.keys()))))
        target_keys = [k for k in common_keys if ".weight" in k and "bias" not in k]
        
        tensors_to_save = {}
        for key in tqdm(target_keys, desc="Comparing"):
            t_tuned = f_tuned.get_tensor(key).float()
            t_base = f_base.get_tensor(key).float()
            
            if t_tuned.shape == t_base.shape:
                delta = t_tuned - t_base
                if torch.abs(delta).max() > threshold:
                    tensors_to_save[key] = delta
            
            del t_tuned, t_base
            aggressive_gc()
            
        for k, v in tensors_to_save.items():
            writer.add_to_index(k, v.shape)
            
        with open(output_lora_path, "wb") as f_out:
            writer.write_header(f_out)
            for key in tqdm(tensors_to_save.keys(), desc="Writing LoRA"):
                writer.write_tensor(f_out, tensors_to_save[key])
                
    print(f"✅ LoRA Extraction complete. Saved to: {output_lora_path}")

def compress_lora(
    input_lora_path: str,
    output_lora_path: str,
    target_rank: int = 64,
    precision: str = "fp16"
) -> None:
    """Compress a full-rank LoRA (or delta) using SVD."""
    print(f"🗜️ Compressing LoRA (Target Rank: {target_rank})")
    writer = SafeTensorsWriter(output_lora_path, precision)
    
    with safe_open(input_lora_path, framework="pt", device="cpu") as f_in:
        keys = sorted(list(f_in.keys()))
        tensors_out = {}
        
        for key in tqdm(keys, desc="SVD Compression"):
            tensor = f_in.get_tensor(key).float()
            orig_shape = tensor.shape
            
            is_conv = len(orig_shape) == 4
            if is_conv:
                tensor = tensor.flatten(start_dim=1)
                
            if min(tensor.shape) <= target_rank:
                tensors_out[key] = tensor.reshape(orig_shape)
            else:
                U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
                U = U[:, :target_rank]
                S = S[:target_rank]
                Vh = Vh[:target_rank, :]
                
                sqrt_s = torch.diag(torch.sqrt(S))
                up = U @ sqrt_s
                down = sqrt_s @ Vh
                
                base_name = key.replace(".weight", "")
                if is_conv:
                    down = down.reshape(target_rank, orig_shape[1], orig_shape[2], orig_shape[3])
                    up = up.reshape(orig_shape[0], target_rank, 1, 1)
                
                tensors_out[f"{base_name}.lora_up.weight"] = up
                tensors_out[f"{base_name}.lora_down.weight"] = down
                tensors_out[f"{base_name}.alpha"] = torch.tensor([float(target_rank)])
                
            del tensor
            aggressive_gc()

        for k, v in tensors_out.items():
            writer.add_to_index(k, v.shape)
            
        with open(output_lora_path, "wb") as f_out:
            writer.write_header(f_out)
            for k in tqdm(tensors_out.keys(), desc="Writing"):
                writer.write_tensor(f_out, tensors_out[k])

    print(f"✅ Compression complete. Saved to: {output_lora_path}")
