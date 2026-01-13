import os
from typing import List, Set, Dict, Optional, Union
from contextlib import ExitStack

import torch
from safetensors import safe_open
from tqdm import tqdm

from .utils import SafeTensorsWriter, aggressive_gc

def ties_op(
    base: torch.Tensor, 
    tensors: List[torch.Tensor], 
    weights: List[float], 
    density: float
) -> torch.Tensor:
    """Core TIES mathematical operation.
    
    Performs Trim, Elect Sign, and Disjoint Merge on a single layer.
    
    Args:
        base: The base model tensor.
        tensors: List of fine-tuned model tensors to merge.
        weights: List of weights corresponding to each model.
        density: The fraction of top-magnitude deltas to keep (0.0 to 1.0).
        
    Returns:
        The merged tensor.
    """
    # 1. Calculate Deltas (Changes from Base) & Apply Weights
    deltas = []
    for t, w in zip(tensors, weights):
        d = (t.float() - base.float()) * w
        deltas.append(d)

    # 2. TRIM (Pruning Noise)
    if density < 1.0:
        for i in range(len(deltas)):
            d_flat = deltas[i].abs().view(-1)
            k = int(d_flat.numel() * (1 - density))
            if k > 0:
                # Find threshold for magnitude pruning
                thresh_val, _ = torch.kthvalue(d_flat, k)
                deltas[i] = torch.where(
                    deltas[i].abs() >= thresh_val, 
                    deltas[i], 
                    torch.tensor(0.0, device=deltas[i].device)
                )

    stacked_deltas = torch.stack(deltas)

    # 3. ELECT SIGN (Voting Direction)
    sum_deltas = torch.sum(stacked_deltas, dim=0)
    majority_sign = torch.sign(sum_deltas)

    # 4. DISJOINT MERGE
    # Keep only deltas that align with the majority sign
    to_keep = (torch.sign(stacked_deltas) == majority_sign)
    final_deltas = stacked_deltas * to_keep

    # 5. RECONSTRUCT (Mean of remaining deltas)
    # Average deltas that survived at each position
    count = torch.sum(to_keep, dim=0).clamp(min=1.0)
    merged_delta = torch.sum(final_deltas, dim=0) / count

    return (base.float() + merged_delta).to(base.dtype)

def ties_merge(
    base_model_path: str,
    models_paths: List[str],
    weights: List[float],
    output_path: str,
    density: float = 0.5,
    precision: str = "fp16"
) -> None:
    """Perform TIES merging on multiple models and save the result.
    
    Args:
        base_model_path: Path to the base model safetensors.
        models_paths: List of paths to fine-tuned model safetensors.
        weights: List of blending weights for each model.
        output_path: Path to save the merged model.
        density: Pruning density (0.0 to 1.0). Default 0.5.
        precision: Output precision ('fp16', 'fp8', 'fp32').
        
    Raises:
        FileNotFoundError: If any input model path does not exist.
        ValueError: If weights and models_paths lengths do not match.
    """
    if len(models_paths) != len(weights):
        raise ValueError("Number of models and weights must match.")

    print(f"🔗 Starting TIES Merge (Density: {density})")
    
    writer = SafeTensorsWriter(output_path, precision)
    
    with ExitStack() as stack:
        # 1. SCANNING & INDEXING
        print("🔍 Phase 1: Indexing structure...")
        try:
            f_base = stack.enter_context(safe_open(base_model_path, framework="pt", device="cpu"))
        except Exception as e:
            print(f"❌ Error opening base model: {e}")
            return

        all_keys = sorted(list(f_base.keys()))
        for key in all_keys:
            slice_data = f_base.get_slice(key)
            writer.add_to_index(key, slice_data.get_shape())

        # Open fine-tuned models
        model_handles = []
        model_keys_sets: List[Set[str]] = []
        for p in models_paths:
            if not os.path.exists(p):
                print(f"⚠️ Model not found: {p}. Skipping.")
                continue
            fh = stack.enter_context(safe_open(p, framework="pt", device="cpu"))
            model_handles.append(fh)
            model_keys_sets.append(set(fh.keys()))

        if not model_handles:
            print("❌ No valid fine-tuned models to merge.")
            return

        # 2. STREAMING PROCESSING
        print("🌊 Phase 2: Processing & Writing...")
        with open(output_path, "wb") as f_out:
            writer.write_header(f_out)
            
            for key in tqdm(all_keys, desc="Merging Layers"):
                try:
                    t_base = f_base.get_tensor(key)
                    
                    current_tensors = []
                    current_weights = []
                    
                    for idx, handle in enumerate(model_handles):
                        if key in model_keys_sets[idx]:
                            t_ft = handle.get_tensor(key)
                            if t_ft.shape == t_base.shape:
                                current_tensors.append(t_ft)
                                current_weights.append(weights[idx])
                    
                    if not current_tensors:
                        result = t_base
                    elif len(current_tensors) == 1:
                        # Weighted sum if only one model has the layer
                        delta = (current_tensors[0].float() - t_base.float()) * current_weights[0]
                        result = (t_base.float() + delta).to(t_base.dtype)
                    else:
                        result = ties_op(t_base, current_tensors, current_weights, density)
                    
                    writer.write_tensor(f_out, result)
                    
                    # Cleanup immediately
                    del t_base, current_tensors, result
                    
                except Exception as e:
                    print(f"\n⚠️ Error on layer {key}: {e}. Falling back to base.")
                    writer.write_tensor(f_out, f_base.get_tensor(key))
                
                # Frequent GC for aggressive memory management
                aggressive_gc()

    print(f"✅ TIES Merging complete. Saved to: {output_path}")
