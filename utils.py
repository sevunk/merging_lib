import gc
import json
import struct
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch

def aggressive_gc() -> None:
    """Force aggressive garbage collection to free up memory.
    
    This function clears the GPU cache (if available) and runs the CPU 
    garbage collector multiple times to ensure immediate cleanup.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    gc.collect()

def clean_tensor(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """Sanitize tensor from NaN/Inf and convert to target precision.
    
    Args:
        tensor: The input torch tensor to clean.
        target_dtype: The desired output torch.dtype.
        
    Returns:
        A cleaned and casted torch tensor.
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        # Handle values for fp16/fp8 safety range
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=65500.0, neginf=-65500.0)
    
    return tensor.to(dtype=target_dtype)

def slerp_tensor(t_a: torch.Tensor, t_b: torch.Tensor, alpha: float, dot_threshold: float = 0.9995) -> torch.Tensor:
    """Perform Spherical Linear Interpolation (SLERP) on tensors.
    
    This implementation handles SLERP per-channel for multi-dimensional tensors.
    
    Args:
        t_a: Starting tensor.
        t_b: Target tensor.
        alpha: Interpolation factor (0.0 to 1.0).
        dot_threshold: Threshold for falling back to linear interpolation.
        
    Returns:
        The interpolated tensor.
    """
    t0 = t_a.float()
    t1 = t_b.float()

    original_shape = t0.shape

    # Per-Channel Handling
    if len(original_shape) > 1:
        v0 = t0.flatten(start_dim=1)
        v1 = t1.flatten(start_dim=1)
    else:
        v0 = t0.unsqueeze(0)
        v1 = t1.unsqueeze(0)

    v0_norm = torch.norm(v0, dim=1, keepdim=True)
    v1_norm = torch.norm(v1, dim=1, keepdim=True)

    v0_n = v0 / (v0_norm + 1e-8)
    v1_n = v1 / (v1_norm + 1e-8)

    dot = torch.sum(v0_n * v1_n, dim=1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    # Use linear interpolation where tensors are too close (numerical stability)
    is_parallel = dot > dot_threshold

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta) + 1e-8

    scale0 = torch.sin(theta * (1.0 - alpha)) / sin_theta
    scale1 = torch.sin(theta * alpha) / sin_theta

    scale0 = torch.where(is_parallel, 1.0 - alpha, scale0)
    scale1 = torch.where(is_parallel, alpha, scale1)

    res = scale0 * v0 + scale1 * v1
    return res.reshape(original_shape).to(t_a.dtype)

def get_precision_info(precision: str) -> Tuple[torch.dtype, str, int]:
    """Map precision string to torch dtype and metadata.
    
    Args:
        precision: High-level precision string ('fp16', 'fp32', 'fp8').
        
    Returns:
        A tuple of (torch_dtype, safetensors_dtype_string, bytes_per_element).
        
    Raises:
        ValueError: If an unsupported precision string is provided.
    """
    if precision == "fp8":
        return torch.float8_e4m3fn, "F8_E4M3", 1
    elif precision == "fp32":
        return torch.float32, "F32", 4
    elif precision == "fp16":
        return torch.float16, "F16", 2
    else:
        raise ValueError(f"Unsupported precision: {precision}")

def smart_convert_lora_key(key: str) -> str:
    """Convert Kohya-style LoRA keys to Diffusers-style for matching.
    
    Args:
        key: The LoRA tensor key.
        
    Returns:
        A 'guessed' base key that matches checkpoint keys.
    """
    key = key.replace(".alpha", "").replace(".weight", "")
    for prefix in ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"]:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    # Standardize separator
    key = key.replace("_", ".")
    
    # Common fixes for specific blocks
    known_components = [
        "double_blocks", "single_blocks", "input_blocks", "middle_block", "output_blocks",
        "img_attn", "txt_attn", "img_mod", "txt_mod",
        "time_text_embed", "context_embedder",
        "proj_mlp", "proj_out", "proj_in",
        "norm1", "norm2", "qkv", "to_k", "to_q", "to_v", "to_out"
    ]

    for comp in known_components:
        key = key.replace(f".{comp}.", f".{comp}.") # Placeholder if we needed specific mapping

    while ".." in key:
        key = key.replace("..", ".")
    return key.strip(".")

class SafeTensorsWriter:
    """Handles buffered streaming write of safetensors to disk.
    
    Attributes:
        output_path: Path to the output safetensors file.
        target_dtype: Torch dtype for the output tensors.
        json_dtype: Safetensors dtype string for the header.
        bpe: Bytes per element for the selected precision.
        header_index: Dictionary to store tensor metadata.
        current_offset: Current byte offset in the data section.
    """

    def __init__(self, output_path: str, precision: str = "fp16"):
        """Initialize the SafeTensorsWriter.
        
        Args:
            output_path: File path where the model will be saved.
            precision: Output precision ('fp16', 'fp32', 'fp8').
        """
        self.output_path = output_path
        self.target_dtype, self.json_dtype, self.bpe = get_precision_info(precision)
        self.header_index: Dict[str, Dict[str, Any]] = {}
        self.current_offset = 0

    def add_to_index(self, key: str, shape: Union[List[int], Tuple[int, ...]]) -> None:
        """Add a tensor metadata to the header index and update offset.
        
        Args:
            key: Tensor name.
            shape: Tensor shape.
        """
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        byte_size = num_elements * self.bpe
        
        self.header_index[key] = {
            "dtype": self.json_dtype,
            "shape": list(shape),
            "data_offsets": [self.current_offset, self.current_offset + byte_size]
        }
        self.current_offset += byte_size

    def write_header(self, f_out) -> int:
        """Pack and write the header JSON to the file.
        
        Args:
            f_out: File handle opened in binary write mode.
            
        Returns:
            The size of the header in bytes.
        """
        header_json = json.dumps(self.header_index, separators=(',', ':')).encode('utf-8')
        header_size = len(header_json)
        f_out.write(struct.pack("<Q", header_size))
        f_out.write(header_json)
        return header_size

    def write_tensor(self, f_out, tensor: torch.Tensor) -> None:
        """Clean and write a tensor to the file.
        
        Args:
            f_out: File handle opened in binary write mode.
            tensor: The tensor to write.
        """
        t_out = clean_tensor(tensor, self.target_dtype)
        
        if self.target_dtype == torch.float8_e4m3fn:
            f_out.write(t_out.view(torch.int8).numpy().tobytes())
        else:
            f_out.write(t_out.numpy().tobytes())
