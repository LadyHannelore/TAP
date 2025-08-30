"""
main.py
Entry point for the Enhanced Quantum GAN project.
This file redirects to the enhanced implementation.
"""

import sys
import os

def main():
    """Main entry point with enhanced features."""
    print("🚀 TAP - Quantum GAN with CUDA Acceleration")
    print("=" * 60)
    print("Starting Enhanced Quantum GAN System...")
    print("=" * 60)
    
    # Try to run the enhanced VQC version first
    try:
        print("Loading enhanced VQC system...")
        import main_vqc
        main_vqc.main()
        return
    except Exception as e:
        print(f"Enhanced VQC system not available: {e}")
        print("Falling back to demo system...")
    
    # Fallback to demo
    try:
        import demo_quantum_gan
        demo_quantum_gan.main()
        return
    except Exception as e:
        print(f"Demo system failed: {e}")
    
    # Basic fallback
    print("\n" + "=" * 60)
    print("BASIC SYSTEM INFORMATION")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print("\n✅ Basic system operational!")
        print("For full functionality, ensure all dependencies are installed.")
        print("\nTry running:")
        print("  python demo_quantum_gan.py  (for quick demo)")
        print("  python main_vqc.py         (for full VQC system)")
        
    except Exception as e:
        print(f"System check failed: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
