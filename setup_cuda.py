"""
setup_cuda.py
Script to install CUDA-enabled packages and verify CUDA setup.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False

def check_system_requirements():
    """Check system requirements for CUDA."""
    print("=" * 60)
    print("CHECKING SYSTEM REQUIREMENTS")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended for optimal CUDA support")
    else:
        print("✓ Python version is compatible")
    
    # Check operating system
    os_name = platform.system()
    print(f"Operating System: {os_name}")
    
    if os_name == "Windows":
        print("✓ Windows detected - CUDA installation supported")
    elif os_name == "Linux":
        print("✓ Linux detected - CUDA installation supported")
    elif os_name == "Darwin":
        print("⚠️  macOS detected - Limited CUDA support (Apple Silicon not supported)")
    
    # Check if NVIDIA GPU is available (basic check)
    if os_name == "Windows":
        nvidia_check = run_command("nvidia-smi", "Checking for NVIDIA GPU")
        if not nvidia_check:
            print("⚠️  NVIDIA GPU not detected or nvidia-smi not available")
            print("   CUDA acceleration will not be available")
            return False
    
    return True

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch."""
    print("\n" + "=" * 60)
    print("INSTALLING CUDA-ENABLED PYTORCH")
    print("=" * 60)
    
    # Determine CUDA version
    cuda_versions = {
        "11.8": "cu118",
        "12.1": "cu121", 
        "12.4": "cu124"
    }
    
    print("Available CUDA versions:")
    for version, code in cuda_versions.items():
        print(f"  {version} ({code})")
    
    # Use CUDA 11.8 as default (most compatible)
    cuda_version = "cu118"
    print(f"Using CUDA version: 11.8 ({cuda_version})")
    
    # Install PyTorch with CUDA support
    pytorch_command = f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version}"
    
    success = run_command(pytorch_command, "Installing CUDA-enabled PyTorch")
    
    if success:
        print("✓ PyTorch with CUDA support installed")
    else:
        print("✗ Failed to install PyTorch with CUDA support")
        print("Trying CPU-only PyTorch as fallback...")
        fallback_command = "pip install torch torchvision torchaudio"
        run_command(fallback_command, "Installing CPU-only PyTorch")
    
    return success

def install_cuda_packages():
    """Install additional CUDA packages."""
    print("\n" + "=" * 60)
    print("INSTALLING ADDITIONAL CUDA PACKAGES")
    print("=" * 60)
    
    packages = [
        ("cupy-cuda11x", "GPU-accelerated NumPy operations"),
        ("numba", "JIT compilation with CUDA support"),
        ("pennylane-lightning-gpu", "GPU-accelerated quantum simulations")
    ]
    
    results = []
    
    for package, description in packages:
        print(f"\nInstalling {package} ({description})...")
        success = run_command(f"pip install {package}", f"Installing {package}")
        results.append((package, success))
    
    # Summary
    print("\n" + "=" * 40)
    print("CUDA PACKAGES INSTALLATION SUMMARY")
    print("=" * 40)
    
    for package, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {package}")
    
    return all(result[1] for result in results)

def verify_cuda_installation():
    """Verify CUDA installation and functionality."""
    print("\n" + "=" * 60)
    print("VERIFYING CUDA INSTALLATION")
    print("=" * 60)
    
    verification_code = """
import torch
import numpy as np

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN available:", torch.backends.cudnn.is_available())
    print("Number of CUDA devices:", torch.cuda.device_count())
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Test basic CUDA operations
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print("✓ Basic CUDA tensor operations working")
    except Exception as e:
        print("✗ CUDA tensor operations failed:", e)
    
    # Test CuPy if available
    try:
        import cupy as cp
        x_cp = cp.random.randn(1000, 1000)
        y_cp = cp.random.randn(1000, 1000)
        z_cp = cp.dot(x_cp, y_cp)
        print("✓ CuPy operations working")
    except ImportError:
        print("✗ CuPy not available")
    except Exception as e:
        print("✗ CuPy operations failed:", e)
    
    # Test Numba CUDA if available
    try:
        from numba import cuda
        print("✓ Numba CUDA available")
        print("  CUDA devices:", len(cuda.gpus))
    except ImportError:
        print("✗ Numba CUDA not available")
    except Exception as e:
        print("✗ Numba CUDA failed:", e)
else:
    print("✗ CUDA not available")
"""
    
    # Write verification script
    with open("verify_cuda.py", "w") as f:
        f.write(verification_code)
    
    # Run verification
    success = run_command("python verify_cuda.py", "Running CUDA verification")
    
    # Clean up
    if os.path.exists("verify_cuda.py"):
        os.remove("verify_cuda.py")
    
    return success

def create_cuda_test_script():
    """Create a test script for CUDA functionality."""
    test_script = """
# CUDA Test Script
# Run this script to test CUDA functionality in your project

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def test_cuda_setup():
    print("Testing CUDA setup for TAP project...")
    
    # Test basic CUDA
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Test CUDA utilities
    try:
        from cuda_utils import get_cuda_manager, benchmark_cuda_vs_cpu
        cuda_manager = get_cuda_manager()
        print(f"CUDA Manager initialized: {cuda_manager.get_device()}")
        
        # Run a quick benchmark
        print("Running CUDA benchmark...")
        benchmark_cuda_vs_cpu(batch_size=100, num_curves=50)
        
    except ImportError as e:
        print(f"CUDA utilities not available: {e}")
    
    # Test enhanced GAN
    try:
        from gan import BezierGAN
        gan = BezierGAN(use_cuda_optimizations=True)
        print(f"Enhanced GAN initialized with device: {gan.device}")
        
    except Exception as e:
        print(f"Enhanced GAN test failed: {e}")
    
    # Test quantum generator
    try:
        from quantum_generator import QuantumGenerator
        qgen = QuantumGenerator(use_gpu=True)
        print("Quantum generator with GPU support initialized")
        
    except Exception as e:
        print(f"Quantum generator test failed: {e}")
    
    print("CUDA testing completed!")

if __name__ == "__main__":
    test_cuda_setup()
"""
    
    with open("test_cuda.py", "w") as f:
        f.write(test_script)
    
    print("✓ Created test_cuda.py script")

def main():
    """Main setup function."""
    print("CUDA SETUP SCRIPT FOR TAP PROJECT")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n⚠️  System requirements not fully met.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Install CUDA-enabled PyTorch
    pytorch_success = install_cuda_pytorch()
    
    # Install additional CUDA packages
    cuda_packages_success = install_cuda_packages()
    
    # Verify installation
    verification_success = verify_cuda_installation()
    
    # Create test script
    create_cuda_test_script()
    
    # Final summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    print(f"PyTorch CUDA: {'✓' if pytorch_success else '✗'}")
    print(f"CUDA packages: {'✓' if cuda_packages_success else '✗'}")
    print(f"Verification: {'✓' if verification_success else '✗'}")
    
    if pytorch_success and verification_success:
        print("\n🎉 CUDA setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python test_cuda.py' to test CUDA functionality")
        print("2. Run 'python main.py' to start the enhanced TAP project")
        print("3. Check generated files for CUDA-accelerated results")
    else:
        print("\n⚠️  CUDA setup completed with issues.")
        print("The project will still work but may use CPU-only operations.")
    
    print("\nFor manual installation instructions, see:")
    print("- PyTorch: https://pytorch.org/get-started/locally/")
    print("- CuPy: https://cupy.dev/")
    print("- Numba: https://numba.pydata.org/")

if __name__ == "__main__":
    main()
