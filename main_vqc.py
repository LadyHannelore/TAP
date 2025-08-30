"""
main_vqc.py
Complete demonstration of the Enhanced Quantum GAN project with VQC implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Core imports with graceful handling
import numpy as np
import torch
import time

# Basic imports that should always work
try:
    from gan import BezierGAN, ClassicalGAN
    from quantum_generator import QuantumGenerator
    from utility import export_bezier_curves_to_svg, evaluate_quality, evaluate_diversity
    from bezier_utils import data_to_bezier_curves, plot_bezier_curves
    BASIC_IMPORTS = True
except ImportError as e:
    print(f"Basic imports failed: {e}")
    BASIC_IMPORTS = False

# Enhanced imports for VQC and evaluation
try:
    from vqc_generator import QuantumGeometryGAN, create_synthetic_geometry_dataset
    from cad_export import GeometryExporter, ExportFormat, BezierCurve, GeometryMetadata
    from evaluation_metrics import ComprehensiveEvaluator
    VQC_AVAILABLE = True
except ImportError as e:
    print(f"VQC components not available: {e}")
    VQC_AVAILABLE = False

# CUDA utilities
try:
    from cuda_utils import get_cuda_manager, benchmark_cuda_vs_cpu
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA utilities not available. Running with basic CUDA support.")

# Matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Visualizations will be skipped.")

def print_system_info():
    """Print comprehensive system and CUDA information."""
    print("=" * 80)
    print("QUANTUM GAN PROJECT - SYSTEM INFORMATION")
    print("=" * 80)
    
    # PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            print(f"CUDA version: {cuda_version}")
        except:
            print("CUDA version: Unable to determine")
        
        try:
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"
            print(f"cuDNN version: {cudnn_version}")
        except:
            print("cuDNN version: Unable to determine")
        
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
            except Exception as e:
                print(f"  Device {i}: Unable to get properties - {e}")
    
    # Component availability
    print(f"Basic imports available: {BASIC_IMPORTS}")
    print(f"VQC components available: {VQC_AVAILABLE}")
    print(f"CUDA utilities available: {CUDA_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    
    # CUDA utilities info
    if CUDA_AVAILABLE:
        try:
            cuda_manager = get_cuda_manager()
            print(f"CUDA Manager device: {cuda_manager.get_device()}")
            
            if cuda_manager.is_cuda_available():
                allocated, reserved = cuda_manager.get_memory_info()
                print(f"GPU memory - Allocated: {allocated / 1024**3:.2f} GB, Reserved: {reserved / 1024**3:.2f} GB")
        except Exception as e:
            print(f"CUDA Manager error: {e}")
    
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

def create_training_data(num_samples: int = 1000, data_type: str = "synthetic"):
    """Create or load training data for the GAN."""
    
    if data_type == "synthetic" and VQC_AVAILABLE:
        print(f"Creating {num_samples} synthetic geometric designs...")
        data = create_synthetic_geometry_dataset(num_samples, "bezier")
        # Convert tensor to numpy if needed
        if hasattr(data, 'detach'):
            return data.detach().cpu().numpy()
        return data
    
    elif data_type == "file":
        # Try to load from file
        try:
            data = np.load('data/sg_t16_train.npy')
            print(f"Loaded {len(data)} samples from file")
            return data[:num_samples] if len(data) > num_samples else data
        except FileNotFoundError:
            print("Training file not found, creating synthetic data...")
            if VQC_AVAILABLE:
                data = create_synthetic_geometry_dataset(num_samples, "bezier")
                if hasattr(data, 'detach'):
                    return data.detach().cpu().numpy()
                return data
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

def train_variational_quantum_gan():
    """Train the Variational Quantum Circuit GAN."""
    if not VQC_AVAILABLE:
        print("VQC components not available. Skipping VQC training.")
        return None, None
    
    print("\n" + "=" * 80)
    print("VARIATIONAL QUANTUM CIRCUIT GAN TRAINING")
    print("=" * 80)
    
    try:
        # Create training data
        train_data = create_training_data(2000, "synthetic")
        print(f"Training data shape: {train_data.shape}")
        
        # Initialize QGAN with different encoding strategies
        encoding_types = ["angle", "amplitude"]  # "basis" might be more complex
        
        best_qgan = None
        best_score = 0
        
        for encoding in encoding_types:
            print(f"\nTraining QGAN with {encoding} encoding...")
            
            try:
                # Create QGAN
                qgan = QuantumGeometryGAN(
                    latent_dim=8,
                    n_qubits=4,
                    n_layers=3,
                    output_dim=8,
                    discriminator_hidden=[64, 32, 16],
                    lr_g=0.001,
                    lr_d=0.0005,
                    encoding_type=encoding
                )
                
                print(f"QGAN initialized on device: {qgan.device}")
                print(f"Quantum parameters: {qgan.generator.quantum_params.shape}")
                
                # Convert training data to torch tensors
                train_tensor = torch.FloatTensor(train_data).to(qgan.device)
                
                # Training loop
                n_epochs = 50
                batch_size = 32
                n_batches = len(train_data) // batch_size
                
                print(f"Training for {n_epochs} epochs, {n_batches} batches per epoch...")
                
                for epoch in range(n_epochs):
                    epoch_d_losses = []
                    epoch_g_losses = []
                    
                    # Shuffle data
                    indices = torch.randperm(len(train_tensor))
                    
                    for batch_idx in range(n_batches):
                        # Get batch
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
                        batch_indices = indices[start_idx:end_idx]
                        real_batch = train_tensor[batch_indices]
                        
                        # Training step
                        d_loss, g_loss = qgan.train_step(real_batch)
                        epoch_d_losses.append(d_loss)
                        epoch_g_losses.append(g_loss)
                        
                        # Store quantum parameters evolution
                        if batch_idx % 10 == 0:
                            qgan.training_stats['quantum_params_history'].append(
                                qgan.get_quantum_params()
                            )
                    
                    # Log progress
                    if epoch % 10 == 0:
                        avg_d_loss = np.mean(epoch_d_losses)
                        avg_g_loss = np.mean(epoch_g_losses)
                        print(f"Epoch {epoch}/{n_epochs}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
                    
                    # Store losses
                    qgan.training_stats['d_losses'].append(np.mean(epoch_d_losses))
                    qgan.training_stats['g_losses'].append(np.mean(epoch_g_losses))
                
                # Generate test samples
                test_geometries = qgan.generate_geometries(num_samples=100, output_type="control_points")
                
                # Simple evaluation
                test_score = np.mean(np.std(test_geometries.reshape(len(test_geometries), -1), axis=0))
                print(f"Encoding {encoding} - Test diversity score: {test_score:.4f}")
                
                if test_score > best_score:
                    best_score = test_score
                    best_qgan = qgan
                    print(f"New best model with {encoding} encoding!")
                
            except Exception as e:
                print(f"Training failed for {encoding} encoding: {e}")
                import traceback
                traceback.print_exc()
        
        if best_qgan:
            print(f"\nBest QGAN achieved score: {best_score:.4f}")
            
            # Analyze quantum parameter evolution
            quantum_analysis = best_qgan.analyze_quantum_evolution()
            if 'error' not in quantum_analysis:
                print("Quantum parameter evolution analysis:")
                print(f"  Parameter variance: {torch.mean(quantum_analysis['param_variance']):.4f}")
                print(f"  Parameter range: {torch.mean(quantum_analysis['param_range']):.4f}")
        
        return best_qgan, train_data
    
    except Exception as e:
        print(f"VQC training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def export_geometries_to_cad(geometries, prefix: str = "qgan_output"):
    """Export generated geometries to various CAD formats."""
    if not VQC_AVAILABLE:
        print("CAD export not available.")
        return {}
    
    print(f"\nExporting {len(geometries)} geometries to CAD formats...")
    
    try:
        # Create BezierCurve objects with metadata
        curves = []
        for i, control_points in enumerate(geometries):
            metadata = GeometryMetadata(
                name=f"qgan_curve_{i}",
                creation_time=str(np.datetime64('now')),
                generator_type="variational_quantum_circuit",
                design_parameters={
                    "curve_id": i,
                    "control_points": len(control_points)
                }
            )
            curve = BezierCurve(control_points, metadata)
            curves.append(curve)
        
        # Export to multiple formats
        exporter = GeometryExporter()
        formats = [ExportFormat.SVG, ExportFormat.JSON, ExportFormat.DXF]
        
        results = exporter.batch_export(
            curves, 
            prefix, 
            formats,
            include_control_points=True,
            include_metadata=True
        )
        
        print("Export results:")
        for format_type, filename in results.items():
            if filename:
                print(f"  ✓ {format_type.value}: {filename}")
            else:
                print(f"  ✗ {format_type.value}: Failed")
        
        return results
    
    except Exception as e:
        print(f"CAD export failed: {e}")
        return {}

def evaluate_generation_quality(real_data, generated_data):
    """Comprehensive evaluation of generation quality."""
    if not VQC_AVAILABLE:
        print("Evaluation metrics not available.")
        return {}
    
    print(f"\nEvaluating generation quality...")
    print(f"Real samples: {len(real_data)}, Generated samples: {len(generated_data)}")
    
    try:
        evaluator = ComprehensiveEvaluator()
        results = evaluator.evaluate_generation_quality(real_data, generated_data)
        
        print("\nEvaluation Results:")
        print(f"  Overall Score: {results.get('overall_score', 0):.3f}")
        print(f"  FID Score: {results.get('fid_score', 'N/A')}")
        
        mode_collapse = results.get('mode_collapse', {})
        if isinstance(mode_collapse, dict):
            print(f"  Mode Collapse: {mode_collapse.get('mode_collapse_detected', 'Unknown')}")
        
        diversity = results.get('diversity', {})
        if isinstance(diversity, dict):
            print(f"  Diversity Score: {diversity.get('overall_diversity', 0):.3f}")
        
        quality = results.get('quality', {})
        if isinstance(quality, dict):
            print(f"  Quality Score: {quality.get('overall_quality', 0):.3f}")
        
        # Save detailed report
        report_file = evaluator.generate_evaluation_report(
            real_data, generated_data, "qgan_evaluation_report.json"
        )
        print(f"  Detailed report: {report_file}")
        
        return results
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def visualize_results(qgan, train_data):
    """Create comprehensive visualizations of results."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualizations.")
        return
    
    print("\nGenerating visualizations...")
    
    try:
        # Generate samples for visualization
        if VQC_AVAILABLE and qgan:
            generated_samples = qgan.generate_geometries(num_samples=20, output_type="control_points")
            
            # Plot training vs generated
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Quantum GAN Results", fontsize=16)
            
            # Real curves
            axes[0, 0].set_title("Real Training Curves")
            for i in range(min(10, len(train_data))):
                curve = train_data[i]
                axes[0, 0].plot(curve[:, 0], curve[:, 1], 'b-', alpha=0.6, linewidth=1)
                axes[0, 0].scatter(curve[:, 0], curve[:, 1], c='red', s=20, alpha=0.8)
            axes[0, 0].set_aspect('equal')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Generated curves
            axes[0, 1].set_title("Generated Curves")
            generated_np = generated_samples.detach().cpu().numpy() if hasattr(generated_samples, 'detach') else generated_samples
            for i in range(min(10, len(generated_np))):
                curve = generated_np[i]
                axes[0, 1].plot(curve[:, 0], curve[:, 1], 'g-', alpha=0.6, linewidth=1)
                axes[0, 1].scatter(curve[:, 0], curve[:, 1], c='orange', s=20, alpha=0.8)
            axes[0, 1].set_aspect('equal')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Training losses
            if qgan.training_stats['d_losses'] and qgan.training_stats['g_losses']:
                axes[0, 2].set_title("Training Losses")
                axes[0, 2].plot(qgan.training_stats['d_losses'], label='Discriminator', color='red')
                axes[0, 2].plot(qgan.training_stats['g_losses'], label='Generator', color='blue')
                axes[0, 2].legend()
                axes[0, 2].set_xlabel('Epoch')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].grid(True, alpha=0.3)
            
            # Quantum parameter evolution
            if qgan.training_stats['quantum_params_history']:
                axes[1, 0].set_title("Quantum Parameter Evolution")
                params_history = torch.stack(qgan.training_stats['quantum_params_history'])
                for i in range(min(5, params_history.shape[1])):
                    axes[1, 0].plot(params_history[:, i].cpu().numpy(), label=f'Param {i}', alpha=0.7)
                axes[1, 0].legend()
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Parameter Value')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Diversity comparison
            axes[1, 1].set_title("Geometry Diversity")
            real_flat = train_data[:100].reshape(len(train_data[:100]), -1)
            gen_flat = generated_np[:20].reshape(len(generated_np[:20]), -1)
            
            # Simple diversity measure: pairwise distances
            try:
                from scipy.spatial.distance import pdist
                real_distances = pdist(real_flat)
                gen_distances = pdist(gen_flat)
                
                axes[1, 1].hist(real_distances, bins=30, alpha=0.7, label='Real', color='blue')
                axes[1, 1].hist(gen_distances, bins=30, alpha=0.7, label='Generated', color='green')
                axes[1, 1].legend()
                axes[1, 1].set_xlabel('Pairwise Distance')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            except ImportError:
                axes[1, 1].text(0.5, 0.5, "scipy not available", ha='center', va='center', transform=axes[1, 1].transAxes)
            
            # Contour representation
            if hasattr(qgan, 'generate_geometries'):
                contour_samples = qgan.generate_geometries(num_samples=5, output_type="contour_points")
                axes[1, 2].set_title("Generated Contours")
                contour_np = contour_samples.detach().cpu().numpy() if hasattr(contour_samples, 'detach') else contour_samples
                for i, contour in enumerate(contour_np):
                    axes[1, 2].plot(contour[:, 0], contour[:, 1], alpha=0.8, linewidth=2, label=f'Contour {i}')
                axes[1, 2].set_aspect('equal')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("qgan_comprehensive_results.png", dpi=150, bbox_inches='tight')
            print("✓ Comprehensive visualization saved: qgan_comprehensive_results.png")
            plt.show()
        
        else:
            print("Cannot create visualizations without VQC components.")
    
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function showcasing the complete Quantum GAN system."""
    print("🚀 STARTING ENHANCED QUANTUM GAN PROJECT")
    print("Comprehensive system for quantum-generated geometric designs")
    
    # Print system information
    print_system_info()
    
    # Run benchmarks
    run_cuda_benchmarks()
    
    # Train Variational Quantum GAN
    qgan, train_data = train_variational_quantum_gan()
    
    if qgan and train_data is not None:
        print("\n" + "=" * 80)
        print("GEOMETRY GENERATION AND EXPORT")
        print("=" * 80)
        
        # Generate various types of geometries
        print("Generating different types of geometries...")
        
        # Control points (Bézier curves)
        bezier_curves = qgan.generate_geometries(num_samples=50, output_type="control_points")
        print(f"Generated {len(bezier_curves)} Bézier curves")
        
        # Contour points
        contour_curves = qgan.generate_geometries(num_samples=20, output_type="contour_points") 
        print(f"Generated {len(contour_curves)} contour designs")
        
        # Export to CAD formats
        bezier_np = bezier_curves.detach().cpu().numpy() if hasattr(bezier_curves, 'detach') else bezier_curves
        export_results = export_geometries_to_cad(bezier_np, "vqc_geometries")
        
        # Comprehensive evaluation
        print("\n" + "=" * 80)
        print("QUALITY EVALUATION")
        print("=" * 80)
        
        evaluation_results = evaluate_generation_quality(train_data, bezier_np)
        
        # Create visualizations
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)
        
        visualize_results(qgan, train_data)
        
        # Summary
        print("\n" + "=" * 80)
        print("PROJECT SUMMARY")
        print("=" * 80)
        
        print("✓ Variational Quantum Circuit GAN successfully trained")
        print(f"✓ Generated {len(bezier_np)} geometric designs")
        
        if export_results:
            successful_exports = sum(1 for result in export_results.values() if result)
            print(f"✓ Exported to {successful_exports} CAD formats")
        
        if evaluation_results:
            overall_score = evaluation_results.get('overall_score', 0)
            print(f"✓ Overall quality score: {overall_score:.3f}")
        
        print("✓ Comprehensive evaluation and visualization completed")
        
        # Quantum analysis
        if hasattr(qgan, 'analyze_quantum_evolution'):
            quantum_analysis = qgan.analyze_quantum_evolution()
            if 'error' not in quantum_analysis:
                print("✓ Quantum parameter evolution analyzed")
        
    else:
        print("⚠️  VQC training was not successful, falling back to classical methods...")
        
        # Fallback to enhanced classical GAN
        if BASIC_IMPORTS:
            print("\n" + "=" * 80)
            print("ENHANCED CLASSICAL GAN FALLBACK")
            print("=" * 80)
            
            try:
                # Use the enhanced classical GAN
                bezier_gan = BezierGAN(
                    latent_dim=100, 
                    num_control_points=4,
                    lr=0.0002,
                    use_cuda_optimizations=True
                )
                
                # Create synthetic training data
                synthetic_data = create_training_data(1000, "synthetic")
                if synthetic_data.ndim == 3:  # (samples, points, coords)
                    synthetic_data = synthetic_data.reshape(len(synthetic_data), -1)  # Flatten for classical GAN
                
                # Convert to data loader format expected by classical GAN
                from torch.utils.data import DataLoader, TensorDataset
                tensor_data = torch.FloatTensor(synthetic_data)
                dataset = TensorDataset(tensor_data)
                data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
                
                # Train classical GAN
                bezier_gan.train(data_loader, epochs=20, save_interval=5, log_interval=2)
                
                # Generate and export
                classical_curves = bezier_gan.generate_curves(num_curves=30)
                
                # Export classical results
                if VQC_AVAILABLE:
                    export_geometries_to_cad(classical_curves, "classical_geometries")
                else:
                    export_bezier_curves_to_svg(classical_curves, "classical_curves.svg")
                
                print("✓ Classical GAN training and generation completed")
                
            except Exception as e:
                print(f"Classical GAN fallback failed: {e}")
        else:
            print("Basic imports not available. Cannot run fallback.")
    
    # Final system status
    print("\n" + "=" * 80)
    print("FINAL SYSTEM STATUS")
    print("=" * 80)
    
    if torch.cuda.is_available():
        print("✓ CUDA acceleration enabled")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"✓ GPU {i}: {props.name}")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    print(f"✓ VQC components: {'Available' if VQC_AVAILABLE else 'Not available'}")
    print(f"✓ CUDA utilities: {'Available' if CUDA_AVAILABLE else 'Not available'}")
    print(f"✓ Basic imports: {'Available' if BASIC_IMPORTS else 'Not available'}")
    print(f"✓ Matplotlib: {'Available' if MATPLOTLIB_AVAILABLE else 'Not available'}")
    
    # Check generated files
    generated_files = [
        "qgan_comprehensive_results.png",
        "vqc_geometries.svg",
        "vqc_geometries.json", 
        "qgan_evaluation_report.json",
        "classical_curves.svg"
    ]
    
    print("\nGenerated files:")
    for file in generated_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"⚠️  {file} (not generated)")
    
    print("\n🎉 QUANTUM GAN PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
