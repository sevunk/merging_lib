import os
import json
import struct
import gc
from typing import List, Dict, Optional, Tuple, Any

import torch
from safetensors.torch import safe_open
from tqdm import tqdm

from .utils import SafeTensorsWriter, aggressive_gc, get_precision_info

def detect_architecture(base_path: str) -> Tuple[str, List[str]]:
    """Detect the model architecture based on Diffusers folder contents.
    
    Args:
        base_path: Path to the root Diffusers folder.
        
    Returns:
        A tuple of (architecture_name, list_of_component_folders).
    """
    comps = set(os.listdir(base_path))

    if "transformer" in comps:
        return "FLUX", ["transformer", "vae", "text_encoder", "text_encoder_2"]
    elif "unet" in comps and "text_encoder_2" in comps:
        return "SDXL", ["unet", "vae", "text_encoder", "text_encoder_2"]
    elif "unet" in comps:
        return "SD15", ["unet", "vae", "text_encoder"]
    
    return "UNKNOWN", []

def get_new_key(old_key: str, component: str, arch: str) -> Optional[str]:
    """Map Diffusers internal keys to standard Safetensors keys.
    
    Args:
        old_key: The original key in the component file.
        component: The component folder name.
        arch: The detected architecture.
        
    Returns:
        The mapped key or None if not recognized.
    """
    if arch == "FLUX":
        if component == "transformer": return f"model.diffusion_model.{old_key}"
        elif component == "vae": return f"vae.{old_key}"
        elif component == "text_encoder": return f"text_encoders.clip_l.transformer.{old_key}"
        elif component == "text_encoder_2": return f"text_encoders.t5xxl.transformer.{old_key}"

    elif arch == "SDXL":
        if component == "unet": return f"model.diffusion_model.{old_key}"
        elif component == "vae": return f"first_stage_model.{old_key}"
        elif component == "text_encoder": return f"conditioner.embedders.0.transformer.{old_key}"
        elif component == "text_encoder_2": return f"conditioner.embedders.1.model.{old_key}"

    elif arch == "SD15":
        if component == "unet": return f"model.diffusion_model.{old_key}"
        elif component == "vae": return f"first_stage_model.{old_key}"
        elif component == "text_encoder": return f"cond_stage_model.transformer.{old_key}"

    return None

def convert_diffusers(
    input_diffusers_path: str,
    output_path: str,
    precision: str = "fp16"
) -> None:
    """Convert a Diffusers model folder to a single .safetensors file.
    
    Args:
        input_diffusers_path: Root folder of the Diffusers model.
        output_path: Target path for the .safetensors file.
        precision: Output precision.
    """
    if not os.path.exists(input_diffusers_path):
        print(f"❌ Input path not found: {input_diffusers_path}")
        return

    arch_type, components = detect_architecture(input_diffusers_path)
    if arch_type == "UNKNOWN":
        print(f"❌ Architecture not recognized at {input_diffusers_path}")
        return

    print(f"🧠 Detected Architecture: {arch_type} | Target Precision: {precision}")
    writer = SafeTensorsWriter(output_path, precision)
    
    # 1. SCANNING PHASE
    print("🔍 Phase 1: Indexing components...")
    file_map = []
    for comp in components:
        comp_path = os.path.join(input_diffusers_path, comp)
        if os.path.exists(comp_path):
            for f in sorted(os.listdir(comp_path)):
                if f.endswith(".safetensors"):
                    file_map.append((os.path.join(comp_path, f), comp))

    if not file_map:
        print("❌ No .safetensors files found in component subfolders.")
        return

    for f_path, comp_type in tqdm(file_map, desc="Indexing"):
        try:
            with safe_open(f_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    new_key = get_new_key(key, comp_type, arch_type)
                    if new_key:
                        writer.add_to_index(new_key, f.get_slice(key).get_shape())
        except Exception as e:
            print(f"❌ Error indexing {f_path}: {e}")
            return

    # 2. STREAMING CONVERSION
    print("🌊 Phase 2: Streaming Conversion...")
    with open(output_path, "wb") as f_out:
        writer.write_header(f_out)
        for f_path, comp_type in tqdm(file_map, desc="Converting"):
            with safe_open(f_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    new_key = get_new_key(key, comp_type, arch_type)
                    if new_key:
                        tensor = f.get_tensor(key)
                        writer.write_tensor(f_out, tensor)
                        del tensor
            aggressive_gc()

    print(f"✅ Conversion complete. Saved to: {output_path}")

def convert_precision(
    input_path: str,
    output_path: str,
    precision: str = "fp16"
) -> None:
    """Convert any .safetensors file to a different precision.
    
    Args:
        input_path: Input .safetensors path.
        output_path: Output .safetensors path.
        precision: Target precision.
    """
    print(f"🔄 Converting precision of {os.path.basename(input_path)} to {precision}")
    writer = SafeTensorsWriter(output_path, precision)
    
    with safe_open(input_path, framework="pt", device="cpu") as f:
        all_keys = sorted(list(f.keys()))
        for key in all_keys:
            writer.add_to_index(key, f.get_slice(key).get_shape())
            
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            for key in tqdm(all_keys, desc="Converting"):
                tensor = f.get_tensor(key)
                writer.write_tensor(f_out, tensor)
                del tensor
                aggressive_gc()

    print(f"✅ Precision conversion complete. Saved to: {output_path}")
