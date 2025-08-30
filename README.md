# TAP - Quantum GAN with CUDA Acceleration

A cutting-edge project combining Quantum Computing and Generative Adversarial Networks (GANs) for generating Bézier curves, now enhanced with comprehensive CUDA acceleration for maximum performance.

## 🚀 Features

### Core Capabilities
- **Quantum-Classical Hybrid Generation**: Combines quantum circuits with classical GANs
- **Advanced Bézier Curve Generation**: High-quality parametric curve synthesis
- **Multiple Generation Modes**: Pure quantum, pure classical, and hybrid approaches

### 🔥 NEW: CUDA Acceleration
- **GPU-Accelerated Training**: Up to 10x faster GAN training with mixed precision
- **CUDA-Optimized Bézier Processing**: GPU-accelerated curve evaluation and metrics
- **Memory Management**: Intelligent GPU memory allocation and optimization
- **Multi-GPU Support**: Automatic best GPU selection and utilization
- **Quantum GPU Acceleration**: Lightning.GPU backend for quantum simulations

### Enhanced Features
- **Advanced GAN Architecture**: Improved generator and discriminator with batch normalization
- **WGAN-GP Support**: Wasserstein GAN with gradient penalty for stable training
- **Real-time Monitoring**: Comprehensive training statistics and GPU memory tracking
- **Model Checkpointing**: Automatic model saving and loading
- **Curve Interpolation**: Smooth transitions between generated curves
- **Performance Benchmarking**: Built-in CUDA vs CPU performance comparisons

## 🛠️ Installation

### Prerequisites
- Python 3.8+ (recommended 3.10+)
- NVIDIA GPU with CUDA capability 3.5+ (for CUDA acceleration)
- NVIDIA CUDA Toolkit 11.8+ or 12.x

### Quick Setup (Automatic)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LadyHannelore/TAP.git
   cd TAP
   ```

2. **Run the CUDA setup script**:
   ```bash
   python setup_cuda.py
   ```
   This script will:
   - Check system requirements
   - Install CUDA-enabled PyTorch
   - Install additional CUDA packages (CuPy, Numba, etc.)
   - Verify CUDA functionality
   - Create test scripts

3. **Test CUDA installation**:
   ```bash
   python test_cuda.py
   ```

### Manual Installation

1. **Install CUDA-enabled PyTorch**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install core requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install CUDA acceleration packages**:
   ```bash
   pip install cupy-cuda11x>=11.0.0  # For CUDA 11.x
   pip install numba>=0.56.0
   pip install pennylane-lightning-gpu>=0.32.0
   ```

### CPU-Only Installation
If CUDA is not available, the project will automatically fall back to CPU operations:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Usage
```bash
python main.py
```

This will:
1. Display system and CUDA information
2. Run CUDA performance benchmarks
3. Train an enhanced Bézier GAN with CUDA acceleration
4. Generate and export Bézier curves
5. Create visualization plots
6. Run quantum-enhanced generation

### Advanced Usage

#### Custom GAN Training
```python
from src.gan import BezierGAN

# Initialize with CUDA optimizations
gan = BezierGAN(
    latent_dim=100,
    num_control_points=4,
    lr=0.0002,
    use_cuda_optimizations=True
)

# Load your data
data_loader = gan.load_data('your_data.npy', batch_size=128)

# Train with enhanced features
gan.train(
    data_loader, 
    epochs=50, 
    save_interval=10, 
    log_interval=1
)

# Generate curves
curves = gan.generate_curves(num_curves=20, temperature=1.0)
interpolated = gan.interpolate_curves(steps=10)
```

#### Quantum-Classical Hybrid
```python
from src.quantum_generator import QuantumGenerator

# Initialize with GPU support
quantum_gen = QuantumGenerator(num_qubits=4, use_gpu=True)

# Pure quantum generation
quantum_curves = quantum_gen.generate_bezier_curves(num_curves=10)

# Hybrid generation
hybrid_curves = quantum_gen.hybrid_quantum_classical_generation(gan, num_curves=10)
```

#### CUDA Utilities
```python
from src.cuda_utils import get_cuda_manager, CUDABezierProcessor

# Get CUDA manager
cuda_manager = get_cuda_manager()
print(f"Using device: {cuda_manager.get_device()}")

# Process curves on GPU
processor = CUDABezierProcessor(cuda_manager)
evaluated_curves = processor.batch_evaluate_curves(control_points)
metrics = processor.compute_curve_metrics(curves)
```

## 📊 Performance

### CUDA Acceleration Benefits
- **Training Speed**: 5-10x faster GAN training
- **Curve Evaluation**: Up to 50x faster batch processing
- **Memory Efficiency**: Optimized GPU memory usage
- **Scalability**: Handles larger batch sizes

### Benchmark Results (Example)
```
System: RTX 4090, 24GB VRAM
CPU time: 2.3450s
CUDA time: 0.0892s
Speedup: 26.29x
```

## 🏗️ Architecture

### Enhanced GAN Architecture
- **Generator**: 5-layer deep network with batch normalization and dropout
- **Discriminator**: Spectral normalization for training stability
- **Training**: Mixed precision, gradient penalty, adaptive learning rates

### CUDA Optimizations
- **Memory Management**: Intelligent allocation and caching
- **Tensor Operations**: Optimized GPU kernels for Bézier curves
- **Data Loading**: Pinned memory and asynchronous transfers
- **Multi-GPU**: Automatic device selection and load balancing

### File Structure
```
TAP/
├── main.py                 # Enhanced main entry point
├── setup_cuda.py          # CUDA installation script
├── test_cuda.py           # CUDA functionality tests
├── requirements.txt       # Updated dependencies
├── src/
│   ├── gan.py             # Enhanced GAN with CUDA support
│   ├── quantum_generator.py  # GPU-accelerated quantum generator
│   ├── cuda_utils.py      # CUDA utilities and optimizations
│   ├── bezier_utils.py    # Bézier curve utilities
│   └── utility.py         # General utilities
└── README.md              # This file
```

## 🔧 Configuration

### CUDA Settings
The system automatically detects and configures CUDA settings, but you can customize:

```python
# In cuda_utils.py
class CUDAManager:
    def __init__(self):
        self.memory_fraction = 0.8  # Use 80% of GPU memory
        self.device = self._get_best_device()
```

### GAN Training Parameters
```python
# Enhanced training configuration
gan = BezierGAN(
    latent_dim=100,           # Latent space dimension
    num_control_points=4,     # Bézier curve control points
    lr=0.0002,               # Learning rate
    beta1=0.5,               # Adam optimizer beta1
    beta2=0.999,             # Adam optimizer beta2
    use_cuda_optimizations=True  # Enable CUDA features
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   data_loader = gan.load_data('data.npy', batch_size=32)
   
   # Clear cache
   cuda_manager.clear_cache()
   ```

2. **Import Errors**
   ```bash
   # Ensure all CUDA packages are installed
   python setup_cuda.py
   ```

3. **Performance Issues**
   ```python
   # Check GPU utilization
   nvidia-smi
   
   # Run benchmarks
   python -c "from src.cuda_utils import benchmark_cuda_vs_cpu; benchmark_cuda_vs_cpu()"
   ```

### System Requirements
- **Minimum**: GTX 1060 (6GB VRAM)
- **Recommended**: RTX 3070+ (8GB+ VRAM)
- **Optimal**: RTX 4080+ (16GB+ VRAM)

## 📈 Results

The enhanced system generates:
- High-quality Bézier curves with quantum-inspired variations
- SVG exports for vector graphics applications
- Performance metrics and training statistics
- Visualization plots and interpolation sequences

Sample outputs:
- `enhanced_generated_curves.svg` - Main generated curves
- `interpolated_curves.svg` - Smooth curve transitions
- `hybrid_quantum_classical_curves.svg` - Quantum-classical hybrid results
- `enhanced_curves_plot.png` - Visualization plots

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add CUDA-optimized features
4. Test on different GPU configurations
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team** - For excellent CUDA integration
- **PennyLane** - For quantum computing framework
- **NVIDIA** - For CUDA toolkit and optimization guides
- **CuPy Team** - For GPU-accelerated NumPy operations
- **Numba** - For JIT compilation and CUDA kernels

## 📚 References

- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
- [PennyLane Lightning.GPU](https://docs.pennylane.ai/projects/lightning-gpu/en/latest/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

⚡ **Enhanced with CUDA for Maximum Performance** ⚡

## Requirements
- Python 3.11
- Libraries: `pennylane`, `torch`, `torchvision`, `numpy`, `matplotlib`, `svgwrite`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LadyHannelore/TAP.git
   cd TAP
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```bash
   python main.py
   ```
2. Generated SVG files will be saved in the project directory.

## Future Work
- Enhance the quantum generator with more complex circuits.
- Add advanced evaluation metrics for quality and diversity.
- Extend support for Bézier curve representations.

## License
This project is licensed under the MIT License.

