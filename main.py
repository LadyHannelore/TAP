"""
main.py
This is the entry point for the Quantum GAN project with Bézier curves.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from gan import BezierGAN, ClassicalGAN
from quantum_generator import QuantumGenerator
from utility import export_bezier_curves_to_svg, evaluate_quality, evaluate_diversity
from bezier_utils import data_to_bezier_curves, plot_bezier_curves
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Starting Quantum GAN Project for Bézier Curves")
    
    # Option 1: Train and use the classical Bézier GAN
    print("\n=== Classical Bézier GAN ===")
    try:
        bezier_gan = BezierGAN(latent_dim=100, num_control_points=4)
        
        # Load and train on your data
        data_loader = bezier_gan.load_data('data/sg_t16_train.npy')
        print(f"Loaded data with {len(data_loader)} batches")
        
        # Train for a few epochs (adjust as needed)
        bezier_gan.train(data_loader, epochs=5)
        
        # Generate new curves
        generated_curves = bezier_gan.generate_curves(num_curves=10)
        print(f"Generated {len(generated_curves)} Bézier curves")
        
        # Export to SVG
        export_bezier_curves_to_svg(generated_curves, "generated_bezier_curves.svg")
        
        # Evaluate quality and diversity
        quality_result = evaluate_quality(generated_curves)
        diversity_result = evaluate_diversity(generated_curves)
        print(quality_result)
        print(diversity_result)
        
        # Plot the curves
        from bezier_utils import BezierCurve
        curve_objects = [BezierCurve(curve) for curve in generated_curves]
        fig, ax = plot_bezier_curves(curve_objects[:5], "Generated Bézier Curves")  # Plot first 5
        plt.savefig("generated_curves_plot.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error with Bézier GAN: {e}")
        # Fallback to legacy classical GAN
        print("Falling back to legacy classical GAN...")
        gan = ClassicalGAN()
        gan.train_legacy(epochs=50)
    
    # Option 2: Use the quantum generator
    print("\n=== Quantum Generator ===")
    try:
        quantum_gen = QuantumGenerator()
        quantum_gen.generate()
    except Exception as e:
        print(f"Error with quantum generator: {e}")
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()
