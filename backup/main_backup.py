"""
main.py
Enhanced Quantum GAN project with comprehensive CUDA acceleration and VQC implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Core imports
from gan import BezierGAN, ClassicalGAN
from quantum_generator import QuantumGenerator
from utility import export_bezier_curves_to_svg, evaluate_quality, evaluate_diversity
from bezier_utils import data_to_bezier_curves, plot_bezier_curves
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# Enhanced imports for VQC and evaluation
try:
    from vqc_generator import QuantumGeometryGAN, create_synthetic_geometry_dataset
    from cad_export import GeometryExporter, ExportFormat, BezierCurve, GeometryMetadata
    from evaluation_metrics import ComprehensiveEvaluator
    VQC_AVAILABLE = True
except ImportError as e:
    print(f"VQC components not available: {e}")
    VQC_AVAILABLE = False
    # Create fallback BezierCurve class
    class BezierCurve:
        def __init__(self, control_points):
            self.control_points = control_points

# CUDA utilities
try:
    from cuda_utils import get_cuda_manager, benchmark_cuda_vs_cpu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA utilities not available. Running with basic CUDA support.")

def print_system_info():
    """Print comprehensive system and CUDA information."""
    print("=" * 80)
    print("QUANTUM GAN PROJECT - SYSTEM INFORMATION")
    print("=" * 80)
    
    # PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        print(f"CUDA version: {cuda_version}")
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"
        print(f"cuDNN version: {cudnn_version}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    
    # VQC and advanced features
    print(f"VQC components available: {VQC_AVAILABLE}")
    
    # CUDA utilities info
    if CUDA_AVAILABLE:
        cuda_manager = get_cuda_manager()
        print(f"CUDA Manager device: {cuda_manager.get_device()}")
        
        if cuda_manager.is_cuda_available():
            allocated, reserved = cuda_manager.get_memory_info()
            print(f"GPU memory - Allocated: {allocated / 1024**3:.2f} GB, Reserved: {reserved / 1024**3:.2f} GB")
    
    print("=" * 80)

def run_cuda_benchmarks():
    """Run comprehensive CUDA performance benchmarks."""
    if not CUDA_AVAILABLE:
        print("CUDA utilities not available. Skipping benchmarks.")
        return
    
    print("\n" + "=" * 80)
    print("CUDA PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    try:
        # Benchmark CUDA vs CPU for Bézier operations
        print("Benchmarking Bézier curve operations...")
        benchmark_cuda_vs_cpu(batch_size=500, num_curves=100)
        
        # Additional GPU memory benchmark
        if torch.cuda.is_available():
            print("\nGPU Memory Benchmark:")
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            print(f"Total GPU Memory: {total_memory / 1024**3:.2f} GB")
            
            # Test memory allocation
            try:
                test_tensor = torch.randn(10000, 10000, device='cuda')
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"Test allocation successful: {allocated:.2f} GB")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Memory test failed: {e}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    print("=" * 80)

def create_training_data(num_samples: int = 1000, data_type: str = "synthetic") -> np.ndarray:
    """Create or load training data for the GAN."""
    
    if data_type == "synthetic" and VQC_AVAILABLE:
        print(f"Creating {num_samples} synthetic geometric designs...")
        return create_synthetic_geometry_dataset(num_samples, "bezier")
    
    elif data_type == "file":
        # Try to load from file
        try:
            data = np.load('data/sg_t16_train.npy')
            print(f"Loaded {len(data)} samples from file")
            return data[:num_samples] if len(data) > num_samples else data
        except FileNotFoundError:
            print("Training file not found, creating synthetic data...")
            if VQC_AVAILABLE:
                return create_synthetic_geometry_dataset(num_samples, "bezier")
            else:
                # Fallback synthetic data
                return np.random.randn(num_samples, 4, 2) * 30 + 50
    
    else:
        # Simple fallback
        print("Creating simple synthetic data...")
        data = []
        for i in range(num_samples):
            # Create simple curves
            t = i / num_samples * 2 * np.pi
            center = np.array([50, 50])
            radius = 20 + 10 * np.sin(3 * t)
            
            control_points = np.array([
                center + radius * np.array([np.cos(t), np.sin(t)]),
                center + radius * np.array([np.cos(t + 0.5), np.sin(t + 0.5)]),
                center + radius * np.array([np.cos(t + 1.0), np.sin(t + 1.0)]),
                center + radius * np.array([np.cos(t + 1.5), np.sin(t + 1.5)])
            ])
            data.append(control_points)
        
        return np.array(data)

def main():
    print("Starting Enhanced Quantum GAN Project for Bézier Curves with CUDA Acceleration")
    
    # Print system information
    print_system_info()
    
    # Run benchmarks
    run_cuda_benchmarks()
    
    # Option 1: Train and use the enhanced classical Bézier GAN with CUDA
    print("\n" + "=" * 60)
    print("ENHANCED CLASSICAL BÉZIER GAN WITH CUDA")
    print("=" * 60)
    try:
        # Initialize with CUDA optimizations
        bezier_gan = BezierGAN(
            latent_dim=100, 
            num_control_points=4,
            lr=0.0002,
            use_cuda_optimizations=True
        )
        
        # Load and train on your data
        try:
            data_loader = bezier_gan.load_data('data/sg_t16_train.npy', batch_size=128)
            print(f"Loaded data with {len(data_loader)} batches")
            
            # Train for a few epochs with enhanced features
            start_time = time.time()
            bezier_gan.train(data_loader, epochs=10, save_interval=5, log_interval=1)
            training_time = time.time() - start_time
            
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Get training statistics
            stats = bezier_gan.get_training_stats()
            print(f"Final Generator Loss: {stats['g_losses'][-1]:.4f}")
            print(f"Final Discriminator Loss: {stats['d_losses'][-1]:.4f}")
            
        except FileNotFoundError:
            print("Training data not found. Generating synthetic data for demonstration...")
            # Create synthetic training data
            synthetic_data = np.random.randn(1000, 8).astype(np.int8)  # 4 control points * 2 coordinates
            np.save('synthetic_train.npy', synthetic_data)
            
            data_loader = bezier_gan.load_data('synthetic_train.npy', batch_size=64)
            bezier_gan.train(data_loader, epochs=5, save_interval=2, log_interval=1)
        
        # Generate new curves
        print("\nGenerating curves...")
        generated_curves = bezier_gan.generate_curves(num_curves=20, temperature=1.0)
        print(f"Generated {len(generated_curves)} Bézier curves")
        
        # Generate interpolated curves
        print("Generating interpolated curves...")
        interpolated_curves = bezier_gan.interpolate_curves(steps=10)
        print(f"Generated {len(interpolated_curves)} interpolated curves")
        
        # Evaluate model performance
        print("Evaluating model performance...")
        performance_metrics = bezier_gan.evaluate_model_performance()
        print("Performance metrics:", performance_metrics)
        
        # Export to SVG
        export_bezier_curves_to_svg(generated_curves, "enhanced_generated_curves.svg")
        export_bezier_curves_to_svg(interpolated_curves, "interpolated_curves.svg")
        
        # Evaluate quality and diversity
        quality_result = evaluate_quality(generated_curves)
        diversity_result = evaluate_diversity(generated_curves)
        print("Quality evaluation:", quality_result)
        print("Diversity evaluation:", diversity_result)
        
        # Plot the curves
        from bezier_utils import BezierCurve
        curve_objects = [BezierCurve(curve) for curve in generated_curves[:8]]  # Plot first 8
        fig, ax = plot_bezier_curves(curve_objects, "Enhanced Generated Bézier Curves")
        plt.savefig("enhanced_curves_plot.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save final model
        bezier_gan.save_checkpoint("final_model.pth")
        
    except Exception as e:
        print(f"Error with enhanced Bézier GAN: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to legacy classical GAN
        print("Falling back to legacy classical GAN...")
        try:
            gan = ClassicalGAN()
            gan.train_legacy(epochs=50)
        except Exception as e2:
            print(f"Error with legacy GAN: {e2}")
    
    # Option 2: Use the enhanced quantum generator
    print("\n" + "=" * 60)
    print("ENHANCED QUANTUM GENERATOR WITH GPU ACCELERATION")
    print("=" * 60)
    try:
        # Initialize quantum generator with GPU support
        quantum_gen = QuantumGenerator(num_qubits=4, use_gpu=torch.cuda.is_available())
        
        # Benchmark quantum performance
        quantum_gen.benchmark_quantum_performance(num_trials=50)
        
        # Generate pure quantum curves
        quantum_curves = quantum_gen.generate(num_curves=10)
        print(f"Generated {len(quantum_curves)} pure quantum curves")
        
        # Hybrid quantum-classical generation
        if 'bezier_gan' in locals():
            print("Generating hybrid quantum-classical curves...")
            hybrid_curves = quantum_gen.hybrid_quantum_classical_generation(
                bezier_gan, num_curves=10
            )
            print(f"Generated {len(hybrid_curves)} hybrid curves")
            
            # Export hybrid curves
            export_bezier_curves_to_svg(hybrid_curves, "hybrid_quantum_classical_curves.svg")
            
            # Plot hybrid curves
            hybrid_curve_objects = [BezierCurve(curve) for curve in hybrid_curves[:5]]
            fig, ax = plot_bezier_curves(hybrid_curve_objects, "Hybrid Quantum-Classical Curves")
            plt.savefig("hybrid_curves_plot.png", dpi=150, bbox_inches='tight')
            plt.show()
        
    except Exception as e:
        print(f"Error with enhanced quantum generator: {e}")
        import traceback
        traceback.print_exc()
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✓ CUDA acceleration enabled")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"✓ Using GPU {i}: {props.name}")
    else:
        print("✗ CUDA not available, using CPU")
    
    if CUDA_AVAILABLE:
        print("✓ Advanced CUDA utilities available")
        print("✓ CUDA-optimized Bézier curve processing")
        print("✓ GPU memory management")
    else:
        print("✗ Advanced CUDA utilities not available")
    
    # Check generated files
    generated_files = [
        "enhanced_generated_curves.svg",
        "interpolated_curves.svg", 
        "enhanced_curves_plot.png",
        "final_model.pth"
    ]
    
    print("\nGenerated files:")
    for file in generated_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")
    
    print("\nProject completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
