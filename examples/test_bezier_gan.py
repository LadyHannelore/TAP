"""
test_bezier_gan.py
Test script for the Bézier GAN implementation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
from src.bezier_utils import BezierCurve, data_to_bezier_curves, plot_bezier_curves
from src.utility import export_bezier_curves_to_svg, evaluate_quality, evaluate_diversity
import matplotlib.pyplot as plt

def test_bezier_functionality():
    print("Testing Bézier curve functionality...")
    
    # Create some test control points for Bézier curves
    test_curves_data = [
        [[0, 0], [30, 60], [70, 60], [100, 0]],  # S-curve
        [[10, 10], [20, 80], [80, 80], [90, 10]],  # Another S-curve
        [[0, 50], [50, 0], [50, 100], [100, 50]],  # Loop-like curve
    ]
    
    # Create BezierCurve objects
    curves = [BezierCurve(points) for points in test_curves_data]
    
    # Test plotting
    print("Plotting test curves...")
    fig, ax = plot_bezier_curves(curves, "Test Bézier Curves")
    plt.savefig("test_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test SVG export
    print("Testing SVG export...")
    export_bezier_curves_to_svg(curves, "test_curves.svg")
    
    # Test evaluation functions
    print("Testing evaluation functions...")
    quality_result = evaluate_quality(curves)
    diversity_result = evaluate_diversity(curves)
    print(quality_result)
    print(diversity_result)
    
    print("Basic functionality test completed!")

def test_data_conversion():
    print("\nTesting data conversion...")
    
    # Load a small sample of your actual data
    try:
        data = np.load('data/sg_t16_train.npy')
        print(f"Loaded data shape: {data.shape}")
        
        # Convert first few data points to Bézier curves
        sample_data = data[:32]  # 32 points = 4 curves with 4 control points each
        print(f"Sample data: {sample_data}")
        
        # Try to convert to Bézier curves
        curves = data_to_bezier_curves(sample_data, num_control_points=4)
        print(f"Converted to {len(curves)} Bézier curves")
        
        if curves:
            # Plot the first curve
            fig, ax = plot_bezier_curves([curves[0]], "Converted Curve from Data")
            plt.savefig("converted_curve.png", dpi=150, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"Error testing data conversion: {e}")

if __name__ == "__main__":
    test_bezier_functionality()
    test_data_conversion()
