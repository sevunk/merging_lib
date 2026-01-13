import os
import json
import struct
import re
from typing import Dict, List, Set, Optional, Tuple, Union

import torch
from safetensors import safe_open
from tqdm import tqdm

from .utils import SafeTensorsWriter, aggressive_gc, smart_convert_lora_key

def get_clean_key(key: str) -> str:
    """Standardize checkpoint keys by removing common architecture prefixes.
    
    Args:
        key: The original tensor key.
        
    Returns:
        The cleaned key.
    """
    prefixes = [
        "model.diffusion_model.",
        "first_stage_model.model.diffusion_model.",
        "transformer.",
        "unet."
    ]
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key

def get_clean_vae_key(key: str) -> str:
    """Standardize VAE keys for cross-format matching.
    
    Args:
        key: The original VAE key.
        
    Returns:
        The cleaned VAE key.
    """
    key = key.replace("first_stage_model.", "").replace("vae.", "").replace("ae.", "")
    key = key.replace("model.", "")
    return key


def bake_lora(
    ckpt_path: str,
    lora_path: str,
    output_path: str,
    strength: float = 1.0,
    precision: str = "fp16"
) -> None:
    """Bake (merge) a LoRA into a checkpoint.
    
    Args:
        ckpt_path: Path to the base checkpoint.
        lora_path: Path to the LoRA safetensors.
        output_path: Path to save the baked model.
        strength: Merging strength multiplier. Default 1.0.
        precision: Output precision ('fp16', 'fp8', 'fp32').
    """
    print(f"🍪 Starting LoRA Baking (Strength: {strength})")
    writer = SafeTensorsWriter(output_path, precision)
    
    with safe_open(ckpt_path, framework="pt", device="cpu") as f_ckpt, \
         safe_open(lora_path, framework="pt", device="cpu") as f_lora:
        
        # 1. Map Checkpoint Keys
        ckpt_keys = sorted(list(f_ckpt.keys()))
        clean_to_real_ckpt = {get_clean_key(k): k for k in ckpt_keys}
        
        for k in ckpt_keys:
            writer.add_to_index(k, f_ckpt.get_slice(k).get_shape())
            
        # 2. Map LoRA Blocks
        lora_map: Dict[str, Dict[str, str]] = {}
        lora_keys = list(f_lora.keys())
        
        for k in lora_keys:
            l_type = None
            if "lora_up" in k: l_type = "up"
            elif "lora_down" in k: l_type = "down"
            elif "alpha" in k: l_type = "alpha"
            else: continue
            
            base_guess = smart_convert_lora_key(k)
            if base_guess not in lora_map:
                lora_map[base_guess] = {}
            lora_map[base_guess][l_type] = k

        # 3. Bake and Write
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            baked_count = 0
            
            for key in tqdm(ckpt_keys, desc="Baking LoRA"):
                t_base = f_ckpt.get_tensor(key)
                clean_k = get_clean_key(key)
                target_lora_k = clean_k.replace(".weight", "")
                
                did_bake = False
                if target_lora_k in lora_map and key.endswith(".weight"):
                    match_info = lora_map[target_lora_k]
                    if "up" in match_info and "down" in match_info:
                        try:
                            up_w = f_lora.get_tensor(match_info["up"]).float()
                            down_w = f_lora.get_tensor(match_info["down"]).float()
                            rank = down_w.shape[0]
                            
                            scaling = strength
                            if "alpha" in match_info:
                                scaling *= (f_lora.get_tensor(match_info["alpha"]).item() / rank)
                                
                            delta = (up_w @ down_w) * scaling
                            if delta.numel() == t_base.numel():
                                result = (t_base.float() + delta.reshape(t_base.shape)).to(t_base.dtype)
                                writer.write_tensor(f_out, result)
                                baked_count += 1
                                did_bake = True
                                del result, delta
                            del up_w, down_w
                        except Exception as e:
                            print(f"⚠️ Failed to bake {key}: {e}")
                
                if not did_bake:
                    writer.write_tensor(f_out, t_base)
                
                del t_base
                aggressive_gc()

    print(f"✅ LoRA Baked! {baked_count} layers modified. Output: {output_path}")

def bake_vae(
    ckpt_path: str,
    vae_path: str,
    output_path: str,
    precision: str = "fp32"
) -> None:
    """Replace VAE blocks in a checkpoint with an external VAE.
    
    Args:
        ckpt_path: Path to the base checkpoint.
        vae_path: Path to the new VAE.
        output_path: Path to save the baked model.
        precision: Output precision ('fp32' recommended for VAE).
    """
    print(f"🍞 Starting VAE Baking")
    writer = SafeTensorsWriter(output_path, precision)
    
    with safe_open(ckpt_path, framework="pt", device="cpu") as f_ckpt, \
         safe_open(vae_path, framework="pt", device="cpu") as f_vae:
        
        vae_keys_map = {get_clean_vae_key(k): k for k in f_vae.keys()}
        ckpt_keys = sorted(list(f_ckpt.keys()))
        
        for k in ckpt_keys:
            clean_k = get_clean_vae_key(k)
            is_vae = any(x in k for x in ["first_stage_model", "vae.", "ae."])
            
            if is_vae and clean_k in vae_keys_map:
                writer.add_to_index(k, f_vae.get_slice(vae_keys_map[clean_k]).get_shape())
            else:
                writer.add_to_index(k, f_ckpt.get_slice(k).get_shape())
                
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            replaced_count = 0
            
            for key in tqdm(ckpt_keys, desc="Baking VAE"):
                clean_k = get_clean_vae_key(key)
                is_vae = any(x in key for x in ["first_stage_model", "vae.", "ae."])
                
                if is_vae and clean_k in vae_keys_map:
                    result = f_vae.get_tensor(vae_keys_map[clean_k])
                    replaced_count += 1
                else:
                    result = f_ckpt.get_tensor(key)
                    
                writer.write_tensor(f_out, result)
                del result
                aggressive_gc()

    print(f"✅ VAE Baked! {replaced_count} layers replaced. Output: {output_path}")
