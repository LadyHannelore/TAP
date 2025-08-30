"""
test_setup.py
Quick test to verify CUDA and quantum setup is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def test_pytorch_cuda():
    """Test PyTorch CUDA functionality."""
    print("Testing PyTorch CUDA...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Device count: {torch.cuda.device_count()}")
            print(f"✓ Current device: {torch.cuda.current_device()}")
            print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
            
            # Test tensor operations
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
            print(f"✓ CUDA tensor operations working: {z.shape}")
            
            return True
        else:
            print("✗ CUDA not available")
            return False
    except Exception as e:
        print(f"✗ PyTorch CUDA test failed: {e}")
        return False

def test_cupy():
    """Test CuPy functionality."""
    print("\nTesting CuPy...")
    try:
        import cupy as cp
        print(f"✓ CuPy imported successfully")
        print(f"✓ CuPy CUDA available: {cp.cuda.is_available()}")
        
        if cp.cuda.is_available():
            # Test array operations
            x = cp.random.randn(100, 100)
            y = cp.random.randn(100, 100)
            z = cp.dot(x, y)
            print(f"✓ CuPy array operations working: {z.shape}")
            return True
        else:
            print("✗ CuPy CUDA not available")
            return False
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")
        return False

def test_pennylane():
    """Test PennyLane quantum functionality."""
    print("\nTesting PennyLane...")
    try:
        import pennylane as qml
        print(f"✓ PennyLane {qml.__version__} imported")
        
        # Test basic quantum device
        dev = qml.device('default.qubit', wires=2)
        
        @qml.qnode(dev)
        def simple_circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        result = simple_circuit(0.5)
        print(f"✓ Basic quantum circuit working: {result}")
        
        # Test if lightning backend is available
        try:
            dev_lightning = qml.device('lightning.qubit', wires=2)
            print("✓ Lightning backend available")
        except:
            print("⚠️  Lightning backend not available, using default.qubit")
        
        return True
    except Exception as e:
        print(f"✗ PennyLane test failed: {e}")
        return False

def test_basic_imports():
    """Test basic scientific computing imports."""
    print("\nTesting basic scientific computing libraries...")
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        
        return True
    except Exception as e:
        print(f"✗ Basic imports test failed: {e}")
        return False

def test_enhanced_components():
    """Test our enhanced quantum GAN components."""
    print("\nTesting enhanced Quantum GAN components...")
    try:
        # Test VQC components
        from vqc_generator import QuantumGeometryGAN, create_synthetic_geometry_dataset
        print("✓ VQC Generator components imported")
        
        # Test CAD export
        from cad_export import GeometryExporter, ExportFormat
        print("✓ CAD Export components imported")
        
        # Test evaluation metrics
        from evaluation_metrics import ComprehensiveEvaluator
        print("✓ Evaluation metrics components imported")
        
        # Test CUDA utilities
        from cuda_utils import CUDAManager, get_cuda_manager
        cuda_manager = get_cuda_manager()
        print(f"✓ CUDA Manager initialized: {cuda_manager.get_device()}")
        
        return True
    except Exception as e:
        print(f"✗ Enhanced components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 QUANTUM GAN PROJECT - SYSTEM VALIDATION")
    print("=" * 60)
    
    results = {
        "PyTorch CUDA": test_pytorch_cuda(),
        "CuPy": test_cupy(), 
        "PennyLane": test_pennylane(),
        "Basic Libraries": test_basic_imports(),
        "Enhanced Components": test_enhanced_components()
    }
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for Quantum GAN training.")
        print("\nNext steps:")
        print("1. Run: python main_vqc.py")
        print("2. Or run individual components for testing")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("The system may still work with reduced functionality.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
