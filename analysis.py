import os
import re
import hashlib
import json
from typing import List, Dict, Optional, Tuple, Any

import torch
from safetensors import safe_open
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def calculate_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file efficiently.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        The SHA256 hex string.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 1MB chunks
        for byte_block in iter(lambda: f.read(1048576), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def inspect_model(model_path: str) -> Dict[str, Any]:
    """Inspect a safetensors file for metadata, hash, and basic structure.
    
    Args:
        model_path: Path to the .safetensors file.
        
    Returns:
        A dictionary containing model information.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    report = {
        "filename": os.path.basename(model_path),
        "size_gb": os.path.getsize(model_path) / (1024**3),
    }

    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            report["total_layers"] = len(f.keys())
            report["metadata"] = f.metadata()
            
            # Peek at first tensor for precision
            if len(f.keys()) > 0:
                first_key = list(f.keys())[0]
                slice_data = f.get_slice(first_key)
                report["precision"] = str(slice_data.get_dtype())
                report["sample_shape"] = slice_data.get_shape()

        print(f"\n🕵️ Model Inspection: {report['filename']}")
        print(f"📦 Size: {report['size_gb']:.2f} GB | Layers: {report.get('total_layers', 0)}")
        print(f"💎 Precision: {report.get('precision', 'Unknown')}")
        
        # Hash is slow, so we print it last
        print("🧮 Calculating SHA256...")
        report["sha256"] = calculate_sha256(model_path)
        print(f"🔑 Hash: {report['sha256']}")
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        report["error"] = str(e)

    return report

def analyze_structure(model_path: str) -> None:
    """Detect and print the architectural structure of a model.
    
    Args:
        model_path: Path to the model.
    """
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    print(f"🩻 X-Ray Report for: {os.path.basename(model_path)}")
    
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        arch = "Unknown"
        blocks = []
        
        # Detection logic
        if any("double_blocks" in k for k in keys):
            arch = "FLUX.1 (DiT)"
            dbl = sorted(list(set([int(re.search(r"double_blocks\.(\d+)", k).group(1)) for k in keys if "double_blocks" in k])))
            sgl = sorted(list(set([int(re.search(r"single_blocks\.(\d+)", k).group(1)) for k in keys if "single_blocks" in k])))
            blocks.append(f"Double Blocks: {len(dbl)} (0-{max(dbl) if dbl else 0})")
            blocks.append(f"Single Blocks: {len(sgl)} (0-{max(sgl) if sgl else 0})")
        elif any("input_blocks" in k for k in keys):
            arch = "SDXL/SD1.5 (U-Net)"
            inp = sorted(list(set([int(re.search(r"input_blocks\.(\d+)", k).group(1)) for k in keys if "input_blocks" in k])))
            out = sorted(list(set([int(re.search(r"output_blocks\.(\d+)", k).group(1)) for k in keys if "output_blocks" in k])))
            blocks.append(f"Input Blocks: {len(inp)} (0-{max(inp) if inp else 0})")
            blocks.append(f"Middle Block: {'Yes' if any('middle_block' in k for k in keys) else 'No'}")
            blocks.append(f"Output Blocks: {len(out)} (0-{max(out) if out else 0})")
        elif any("layers." in k for k in keys):
            arch = "Z-Image / Transformer"
            lyrs = sorted(list(set([int(re.search(r"layers\.(\d+)", k).group(1)) for k in keys if "layers." in k])))
            blocks.append(f"Main Layers: {len(lyrs)} (0-{max(lyrs) if lyrs else 0})")
            
        print(f"🧠 Type: {arch} | Total Tensors: {len(keys)}")
        for b in blocks:
            print(f"🧱 {b}")

@torch.no_grad()
def analyze_similarity(
    path_a: str,
    path_b: str,
    metric: str = "Cosine Similarity",
    plot: bool = True
) -> List[float]:
    """Compare two models layer by layer and optionally plot the results.
    
    Args:
        path_a: Path to first model.
        path_b: Path to second model.
        metric: 'Cosine Similarity' or 'MSE'.
        plot: Whether to generate a matplotlib plot.
        
    Returns:
        List of similarity scores per layer.
    """
    print(f"📊 Analyzing Similarity ({metric})")
    
    with safe_open(path_a, framework="pt", device="cpu") as f_a, \
         safe_open(path_b, framework="pt", device="cpu") as f_b:
        
        keys_a = set(f_a.keys())
        keys_b = set(f_b.keys())
        common_keys = sorted(list(keys_a.intersection(keys_b)))
        target_keys = [k for k in common_keys if ".weight" in k]
        
        scores = []
        for key in tqdm(target_keys, desc="Calculating"):
            t_a = f_a.get_tensor(key).float()
            t_b = f_b.get_tensor(key).float()
            
            if t_a.shape != t_b.shape:
                continue
                
            if metric == "Cosine Similarity":
                v_a = t_a.view(-1)
                v_b = t_b.view(-1)
                sim = torch.dot(v_a, v_b) / (torch.norm(v_a) * torch.norm(v_b) + 1e-8)
                score = sim.item()
            else:
                score = torch.mean((t_a - t_b) ** 2).item()
                
            scores.append(score)
            del t_a, t_b
            
    if plot and plt:
        plt.figure(figsize=(12, 6))
        plt.plot(scores, label=metric, color='cyan', alpha=0.8)
        plt.title(f"Layer Similarity: {os.path.basename(path_a)} vs {os.path.basename(path_b)}")
        plt.xlabel("Layer Index")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    elif plot and not plt:
        print("⚠️ Matplotlib not found. Skipping plot.")
        
    avg = sum(scores) / len(scores) if scores else 0
    print(f"📈 Average {metric}: {avg:.4f}")
    return scores

def visualize_preset(preset_values: List[float], total_blocks: int = 57) -> None:
    """Visualize how an MBW weight preset interpolates across blocks.
    
    Args:
        preset_values: List of weights (e.g. from preset_string).
        total_blocks: Total number of blocks to simulate (e.g. 57 for Flux, 20 for SDXL).
    """
    if not plt:
        print("⚠️ Matplotlib not found. Cannot visualize.")
        return
        
    x = np.linspace(0, total_blocks - 1, len(preset_values))
    x_new = np.arange(total_blocks)
    y_smooth = np.interp(x_new, x, preset_values)
    
    plt.figure(figsize=(10, 4))
    plt.plot(y_smooth, marker='o', linestyle='-', color='cyan', alpha=0.8)
    plt.title(f"MBW Preset Visualization ({total_blocks} Blocks)")
    plt.xlabel("Block Index")
    plt.ylabel("Alpha/Weight")
    plt.grid(True, alpha=0.3)
    plt.show()
