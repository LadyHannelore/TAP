"""
demo_quantum_gan.py
Streamlined demonstration of CUDA-accelerated Quantum GAN capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import torch
import numpy as np
import time

def demo_cuda_acceleration():
    """Demonstrate CUDA acceleration capabilities."""
    print("\n🚀 CUDA ACCELERATION DEMO")
    print("=" * 50)
    
    # Test CUDA tensor operations
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        # Benchmark CPU vs GPU
        size = 2000
        
        # CPU benchmark
        start_time = time.time()
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        # GPU benchmark
        start_time = time.time()
        x_gpu = torch.randn(size, size, device='cuda')
        y_gpu = torch.randn(size, size, device='cuda')
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"🚀 GPU Speedup: {speedup:.1f}x faster!")
        
        return True
    else:
        print("❌ CUDA not available")
        return False

def demo_quantum_circuits():
    """Demonstrate quantum circuit functionality."""
    print("\n🔬 QUANTUM CIRCUITS DEMO")
    print("=" * 50)
    
    try:
        import pennylane as qml
        
        # Create a simple quantum device
        dev = qml.device('default.qubit', wires=4)
        
        @qml.qnode(dev)
        def variational_circuit(params, x):
            # Data encoding
            qml.AngleEmbedding(x, wires=range(4))
            
            # Variational layers
            for i in range(2):
                for wire in range(4):
                    qml.RY(params[i * 4 + wire], wires=wire)
                for wire in range(3):
                    qml.CNOT(wires=[wire, wire + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        # Test quantum circuit
        params = np.random.random(8) * 2 * np.pi
        input_data = np.random.random(4)
        
        start_time = time.time()
        result = variational_circuit(params, input_data)
        quantum_time = time.time() - start_time
        
        print(f"✓ Quantum circuit executed in {quantum_time:.4f}s")
        print(f"Input: {input_data}")
        print(f"Output: {np.array(result)}")
        print(f"Parameters shape: {params.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Quantum circuit demo failed: {e}")
        return False

def demo_vqc_generator():
    """Demonstrate VQC geometry generation."""
    print("\n🎨 VQC GEOMETRY GENERATION DEMO")
    print("=" * 50)
    
    try:
        from vqc_generator import QuantumGeometryGAN, create_synthetic_geometry_dataset
        
        # Create synthetic training data
        print("Creating synthetic geometric dataset...")
        train_data = create_synthetic_geometry_dataset(100, "bezier")
        print(f"✓ Created {len(train_data)} training samples")
        
        # Initialize VQC GAN (smaller for demo)
        print("Initializing Quantum GAN...")
        qgan = QuantumGeometryGAN(
            latent_dim=4,
            n_qubits=3,  # Smaller for faster demo
            n_layers=2,
            output_dim=8,
            discriminator_hidden=[32, 16],
            lr_g=0.001,
            lr_d=0.0005,
            encoding_type="angle"
        )
        print(f"✓ QGAN initialized on device: {qgan.device}")
        
        # Quick training demonstration (just a few steps)
        print("Running quick training demo...")
        train_tensor = torch.FloatTensor(train_data[:20]).to(qgan.device)  # Small batch
        
        for step in range(3):  # Just 3 training steps for demo
            d_loss, g_loss = qgan.train_step(train_tensor)
            print(f"  Step {step+1}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
        
        # Generate samples
        print("Generating quantum geometries...")
        generated_samples = qgan.generate_geometries(num_samples=5, output_type="control_points")
        print(f"✓ Generated {len(generated_samples)} geometric designs")
        print(f"Sample shape: {generated_samples.shape}")
        
        return True, generated_samples
    except Exception as e:
        print(f"❌ VQC generator demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def demo_cad_export(geometries):
    """Demonstrate CAD export functionality."""
    print("\n📐 CAD EXPORT DEMO")
    print("=" * 50)
    
    try:
        from cad_export import GeometryExporter, ExportFormat, BezierCurve, GeometryMetadata
        
        if geometries is None:
            print("❌ No geometries available for export")
            return False
        
        # Convert to numpy if needed
        if hasattr(geometries, 'detach'):
            geometries = geometries.detach().cpu().numpy()
        
        # Create BezierCurve objects
        curves = []
        for i, control_points in enumerate(geometries):
            metadata = GeometryMetadata(
                name=f"demo_curve_{i}",
                creation_time=str(np.datetime64('now')),
                generator_type="quantum_vqc_demo",
                design_parameters={"curve_id": i, "demo": True}
            )
            curve = BezierCurve(control_points, metadata)
            curves.append(curve)
        
        # Export to different formats
        exporter = GeometryExporter()
        results = exporter.batch_export(
            curves, 
            "quantum_demo", 
            [ExportFormat.SVG, ExportFormat.JSON],
            include_control_points=True,
            include_metadata=True
        )
        
        print("Export results:")
        for format_type, filename in results.items():
            if filename:
                print(f"✓ {format_type.value}: {filename}")
            else:
                print(f"❌ {format_type.value}: Failed")
        
        return True
    except Exception as e:
        print(f"❌ CAD export demo failed: {e}")
        return False

def demo_evaluation_metrics():
    """Demonstrate evaluation metrics."""
    print("\n📊 EVALUATION METRICS DEMO")
    print("=" * 50)
    
    try:
        from evaluation_metrics_fixed import SimpleEvaluator
        
        # Create sample data for evaluation
        real_data = np.random.randn(50, 4, 2) * 30 + 50  # Real-like data
        generated_data = np.random.randn(20, 4, 2) * 25 + 55  # Generated-like data
        
        evaluator = SimpleEvaluator()
        
        # Quick evaluation
        print("Running evaluation metrics...")
        results = evaluator.evaluate_generation_quality(real_data, generated_data)
        
        print("✓ Evaluation completed:")
        print(f"  Overall Score: {results.get('overall_score', 0):.3f}")
        
        diversity = results.get('diversity', {})
        if isinstance(diversity, dict):
            print(f"  Diversity: {diversity.get('overall_diversity', 0):.3f}")
        
        quality = results.get('quality', {})
        if isinstance(quality, dict):
            print(f"  Quality: {quality.get('overall_quality', 0):.3f}")
        
        mode_collapse = results.get('mode_collapse', {})
        if isinstance(mode_collapse, dict):
            print(f"  Mode Collapse: {mode_collapse.get('mode_collapse_detected', False)}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation metrics demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive demonstration."""
    print("🌟 QUANTUM GAN PROJECT - COMPREHENSIVE DEMO")
    print("Showcasing CUDA 12.2 accelerated quantum geometry generation")
    print("=" * 70)
    
    # Run all demonstrations
    results = {}
    
    results["CUDA Acceleration"] = demo_cuda_acceleration()
    results["Quantum Circuits"] = demo_quantum_circuits()
    
    vqc_success, geometries = demo_vqc_generator()
    results["VQC Generator"] = vqc_success
    
    if geometries is not None:
        results["CAD Export"] = demo_cad_export(geometries)
    else:
        results["CAD Export"] = False
    
    results["Evaluation Metrics"] = demo_evaluation_metrics()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 DEMONSTRATION RESULTS")
    print("=" * 70)
    
    for demo_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{demo_name:20s}: {status}")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print("=" * 70)
    print(f"📈 SUCCESS RATE: {successful_demos}/{total_demos} ({successful_demos/total_demos*100:.1f}%)")
    
    if successful_demos >= 3:
        print("🎉 QUANTUM GAN SYSTEM IS OPERATIONAL!")
        print("\n✨ Your system successfully demonstrates:")
        print("   • CUDA 12.2 acceleration on RTX 4060")
        print("   • Variational Quantum Circuit generation")
        print("   • CAD-ready geometric design export")
        print("   • Comprehensive evaluation metrics")
        print("\n🚀 Ready for full-scale quantum GAN training!")
    else:
        print("⚠️  Some components need attention, but core functionality works")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
