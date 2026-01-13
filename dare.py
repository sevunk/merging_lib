import os
from typing import List, Set, Dict, Optional, Union
from contextlib import ExitStack

import torch
from safetensors import safe_open
from tqdm import tqdm

from .utils import SafeTensorsWriter, aggressive_gc

def dare_op(
    base: torch.Tensor, 
    ft_tensor: torch.Tensor, 
    drop_rate: float, 
    rescale_factor: float
) -> torch.Tensor:
    """Core DARE mathematical operation.
    
    Args:
        base: The base model tensor.
        ft_tensor: The fine-tuned model tensor.
        drop_rate: Probability of dropping a delta (0.0 to 1.0).
        rescale_factor: Scaling factor to compensate for dropped deltas.
        
    Returns:
        The processed delta tensor.
    """
    # 1. Calculate Delta
    delta = ft_tensor.float() - base.float()
    
    # 2. Random Mask (Bernoulli)
    # 0 = Drop, 1 = Keep. Probability of Keep = 1 - drop_rate
    mask = torch.rand_like(delta) < (1.0 - drop_rate)
    
    # 3. Apply Mask & Rescale
    return delta * mask * rescale_factor

def dare_merge(
    base_model_path: str,
    models_paths: List[str],
    output_path: str,
    drop_rate: float = 0.85,
    seed: int = 42,
    precision: str = "fp16"
) -> None:
    """Perform DARE merging on multiple models.
    
    Args:
        base_model_path: Path to the base model.
        models_paths: List of fine-tuned model paths to merge.
        output_path: Target path for the merged model.
        drop_rate: Fraction of deltas to drop. Default 0.85.
        seed: Random seed for reproducibility. Default 42.
        precision: Output precision ('fp16', 'fp8', 'fp32').
    """
    print(f"🧟 Starting DARE Merge (Drop Rate: {drop_rate})")
    torch.manual_seed(seed)
    
    rescale_factor = 1.0 / (1.0 - drop_rate)
    writer = SafeTensorsWriter(output_path, precision)
    
    with ExitStack() as stack:
        # Indexing
        try:
            f_base = stack.enter_context(safe_open(base_model_path, framework="pt", device="cpu"))
        except Exception as e:
            print(f"❌ Error: {e}")
            return
            
        all_keys = sorted(list(f_base.keys()))
        for key in all_keys:
            writer.add_to_index(key, f_base.get_slice(key).get_shape())
            
        # Open models
        model_handles = []
        model_keys_sets = []
        for p in models_paths:
            if os.path.exists(p):
                fh = stack.enter_context(safe_open(p, framework="pt", device="cpu"))
                model_handles.append(fh)
                model_keys_sets.append(set(fh.keys()))
                
        if not model_handles:
            print("❌ No valid models found.")
            return

        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            
            for key in tqdm(all_keys, desc="DARE Merging"):
                try:
                    t_base = f_base.get_tensor(key)
                    total_delta = torch.zeros_like(t_base, dtype=torch.float32)
                    has_change = False
                    
                    for idx, handle in enumerate(model_handles):
                        if key in model_keys_sets[idx]:
                            t_m = handle.get_tensor(key)
                            if t_m.shape == t_base.shape:
                                # Apply DARE op to each model relative to base
                                delta_m = dare_op(t_base, t_m, drop_rate, rescale_factor)
                                total_delta += delta_m
                                has_change = True
                            del t_m
                    
                    if has_change:
                        result = (t_base.float() + total_delta).to(t_base.dtype)
                    else:
                        result = t_base
                        
                    writer.write_tensor(f_out, result)
                    del t_base, total_delta, result
                    
                except Exception as e:
                    print(f"\n⚠️ Error on layer {key}: {e}")
                    writer.write_tensor(f_out, f_base.get_tensor(key))
                    
                aggressive_gc()

    print(f"✅ DARE Merge complete: {output_path}")
