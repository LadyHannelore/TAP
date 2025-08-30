"""
cuda_utils.py
CUDA utilities and optimizations for the Quantum GAN project.
"""

import torch
import numpy as np
import warnings
from typing import Optional, Tuple, List
import math

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✓ CuPy successfully imported")
except Exception as e:
    print(f"⚠️  CuPy import failed: {e}")
    CUPY_AVAILABLE = False
    # Create dummy cp module to prevent import errors
    class DummyCuPy:
        def __getattr__(self, name):
            def dummy_func(*args, **kwargs):
                raise RuntimeError("CuPy not available")
            return dummy_func
    cp = DummyCuPy()

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
    print("✓ Numba CUDA successfully imported")
except Exception as e:
    print(f"⚠️  Numba CUDA import failed: {e}")
    NUMBA_CUDA_AVAILABLE = False

class CUDAManager:
    """Manages CUDA device selection and memory optimization."""
    
    def __init__(self):
        self.device = self._get_best_device()
        self.memory_fraction = 0.8  # Use 80% of GPU memory by default
        self._setup_memory_management()
    
    def _get_best_device(self) -> torch.device:
        """Get the best available CUDA device."""
        if not torch.cuda.is_available():
            print("CUDA not available. Using CPU.")
            return torch.device("cpu")
        
        # Get device with most memory
        best_device = 0
        max_memory = 0
        
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.get_device_properties(i).total_memory
            if memory > max_memory:
                max_memory = memory
                best_device = i
        
        device = torch.device(f"cuda:{best_device}")
        print(f"Selected CUDA device: {device} ({torch.cuda.get_device_name(best_device)})")
        print(f"Total GPU memory: {max_memory / 1024**3:.2f} GB")
        
        return device
    
    def _setup_memory_management(self):
        """Setup CUDA memory management."""
        if self.device.type == "cuda":
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device.index)
            
            # Enable memory caching
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    
    def get_device(self) -> torch.device:
        """Get the CUDA device."""
        return self.device
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self.device.type == "cuda"
    
    def get_memory_info(self) -> Tuple[int, int]:
        """Get current GPU memory usage."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)
        return 0, 0
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def optimize_tensor(self, tensor: torch.Tensor, pin_memory: bool = True) -> torch.Tensor:
        """Optimize tensor for CUDA operations."""
        if self.device.type == "cuda":
            tensor = tensor.to(self.device, non_blocking=True)
            if pin_memory and tensor.is_cpu:
                tensor = tensor.pin_memory()
        return tensor

# CUDA-optimized Bézier curve operations
@cuda.jit
def cuda_bezier_point(t, control_points, result):
    """CUDA kernel for computing Bézier curve points."""
    idx = cuda.grid(1)
    if idx < result.shape[0]:
        n = control_points.shape[0] - 1
        x, y = 0.0, 0.0
        
        # Compute binomial coefficients and Bézier basis
        for i in range(n + 1):
            # Binomial coefficient
            binom = 1.0
            for j in range(1, i + 1):
                binom *= (n - j + 1) / j
            
            # Bézier basis function
            basis = binom * (t[idx] ** i) * ((1 - t[idx]) ** (n - i))
            
            x += basis * control_points[i, 0]
            y += basis * control_points[i, 1]
        
        result[idx, 0] = x
        result[idx, 1] = y

def cuda_evaluate_bezier_batch(control_points_batch: np.ndarray, 
                              t_values: np.ndarray) -> np.ndarray:
    """Evaluate multiple Bézier curves on GPU using CUDA."""
    if not NUMBA_CUDA_AVAILABLE:
        return cpu_evaluate_bezier_batch(control_points_batch, t_values)
    
    batch_size, num_points, _ = control_points_batch.shape
    num_t = len(t_values)
    
    # Allocate GPU memory
    d_control_points = cuda.to_device(control_points_batch.astype(np.float32))
    d_t_values = cuda.to_device(t_values.astype(np.float32))
    d_result = cuda.device_array((batch_size, num_t, 2), dtype=np.float32)
    
    # Configure grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (num_t + threads_per_block - 1) // threads_per_block
    
    # Evaluate each curve in the batch
    results = np.zeros((batch_size, num_t, 2), dtype=np.float32)
    for i in range(batch_size):
        cuda_bezier_point[blocks_per_grid, threads_per_block](
            d_t_values, d_control_points[i], d_result[i]
        )
        results[i] = d_result[i].copy_to_host()
    
    return results

def cpu_evaluate_bezier_batch(control_points_batch: np.ndarray, 
                             t_values: np.ndarray) -> np.ndarray:
    """CPU fallback for Bézier curve evaluation."""
    batch_size, num_points, _ = control_points_batch.shape
    num_t = len(t_values)
    results = np.zeros((batch_size, num_t, 2))
    
    for batch_idx in range(batch_size):
        control_points = control_points_batch[batch_idx]
        n = len(control_points) - 1
        
        for t_idx, t in enumerate(t_values):
            point = np.zeros(2)
            for i in range(n + 1):
                # Binomial coefficient
                binom = math.comb(n, i)
                # Bézier basis function
                basis = binom * (t ** i) * ((1 - t) ** (n - i))
                point += basis * control_points[i]
            results[batch_idx, t_idx] = point
    
    return results

class CUDABezierProcessor:
    """CUDA-accelerated Bézier curve processor."""
    
    def __init__(self, cuda_manager: CUDAManager):
        self.cuda_manager = cuda_manager
        self.use_cupy = CUPY_AVAILABLE and cuda_manager.is_cuda_available()
    
    def batch_evaluate_curves(self, control_points_batch: np.ndarray, 
                             num_points: int = 100) -> np.ndarray:
        """Evaluate multiple Bézier curves efficiently."""
        t_values = np.linspace(0, 1, num_points)
        
        if self.use_cupy:
            return self._cupy_evaluate_batch(control_points_batch, t_values)
        else:
            return cuda_evaluate_bezier_batch(control_points_batch, t_values)
    
    def _cupy_evaluate_batch(self, control_points_batch: np.ndarray, 
                            t_values: np.ndarray) -> np.ndarray:
        """CuPy implementation for Bézier evaluation."""
        # Transfer to GPU
        cp_control_points = cp.asarray(control_points_batch)
        cp_t_values = cp.asarray(t_values)
        
        batch_size, num_control_points, _ = cp_control_points.shape
        num_t = len(cp_t_values)
        n = num_control_points - 1
        
        # Vectorized Bézier evaluation
        results = cp.zeros((batch_size, num_t, 2))
        
        for i in range(num_control_points):
            # Binomial coefficient using math module (compatible)
            import math
            binom = math.comb(n, i)
            # Bézier basis functions (broadcasted)
            t_expanded = cp_t_values[:, None]  # Shape: (num_t, 1)
            basis = binom * (t_expanded ** i) * ((1 - t_expanded) ** (n - i))
            
            # Add contribution of this control point
            control_point = cp_control_points[:, i:i+1, :]  # Shape: (batch_size, 1, 2)
            contribution = basis[None, :, None] * control_point  # Broadcasting
            results += contribution
        
        return cp.asnumpy(results)
    
    def compute_curve_metrics(self, curves: np.ndarray) -> dict:
        """Compute various metrics for generated curves using GPU acceleration."""
        if self.use_cupy:
            curves_gpu = cp.asarray(curves)
            
            # Compute curve lengths
            derivatives = cp.diff(curves_gpu, axis=1)
            segment_lengths = cp.linalg.norm(derivatives, axis=2)
            curve_lengths = cp.sum(segment_lengths, axis=1)
            
            # Compute curvature (simplified)
            second_derivatives = cp.diff(derivatives, axis=1)
            curvature = cp.linalg.norm(second_derivatives, axis=2)
            max_curvature = cp.max(curvature, axis=1)
            
            return {
                'lengths': cp.asnumpy(curve_lengths),
                'max_curvature': cp.asnumpy(max_curvature),
                'mean_length': float(cp.mean(curve_lengths)),
                'std_length': float(cp.std(curve_lengths))
            }
        else:
            # CPU fallback
            derivatives = np.diff(curves, axis=1)
            segment_lengths = np.linalg.norm(derivatives, axis=2)
            curve_lengths = np.sum(segment_lengths, axis=1)
            
            second_derivatives = np.diff(derivatives, axis=1)
            curvature = np.linalg.norm(second_derivatives, axis=2)
            max_curvature = np.max(curvature, axis=1)
            
            return {
                'lengths': curve_lengths,
                'max_curvature': max_curvature,
                'mean_length': float(np.mean(curve_lengths)),
                'std_length': float(np.std(curve_lengths))
            }

# Global CUDA manager instance
cuda_manager = CUDAManager()

def get_cuda_manager() -> CUDAManager:
    """Get the global CUDA manager instance."""
    return cuda_manager

def benchmark_cuda_vs_cpu(batch_size: int = 1000, num_curves: int = 100):
    """Benchmark CUDA vs CPU performance for Bézier curve operations."""
    print(f"Benchmarking CUDA vs CPU performance...")
    print(f"Batch size: {batch_size}, Curves per batch: {num_curves}")
    
    # Generate random control points
    control_points = np.random.rand(batch_size, 4, 2) * 100
    
    import time
    
    # CPU benchmark
    start_time = time.time()
    cpu_results = cpu_evaluate_bezier_batch(control_points, np.linspace(0, 1, num_curves))
    cpu_time = time.time() - start_time
    
    # CUDA benchmark
    if cuda_manager.is_cuda_available():
        start_time = time.time()
        cuda_results = cuda_evaluate_bezier_batch(control_points, np.linspace(0, 1, num_curves))
        cuda_time = time.time() - start_time
        
        speedup = cpu_time / cuda_time
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"CUDA time: {cuda_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results are similar
        max_diff = np.max(np.abs(cpu_results - cuda_results))
        print(f"Maximum difference between CPU and CUDA results: {max_diff:.6f}")
    else:
        print("CUDA not available for benchmark")
        print(f"CPU time: {cpu_time:.4f}s")

if __name__ == "__main__":
    # Run benchmark
    benchmark_cuda_vs_cpu()
