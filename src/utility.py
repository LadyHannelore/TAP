"""
utility.py
This file contains utility functions for SVG export and Bézier curve handling.
"""

import svgwrite
import numpy as np
from bezier_utils import BezierCurve

def export_bezier_curves_to_svg(curves, filename="bezier_output.svg", canvas_size=(400, 400)):
    """
    Export Bézier curves to SVG file.
    curves: list of BezierCurve objects or numpy arrays of control points
    """
    print(f"Exporting {len(curves)} Bézier curves to {filename}...")
    
    dwg = svgwrite.Drawing(filename, size=canvas_size, profile='tiny')
    
    # Add a background
    dwg.add(dwg.rect(insert=(0, 0), size=canvas_size, fill='white'))
    
    for i, curve in enumerate(curves):
        if isinstance(curve, np.ndarray):
            # Convert numpy array to BezierCurve object
            curve = BezierCurve(curve)
        
        # Get SVG path string
        path_str = curve.to_svg_path()
        
        # Add the curve to the drawing
        dwg.add(dwg.path(d=path_str, stroke='blue', stroke_width=2, fill='none'))
        
        # Optionally add control points
        for j, cp in enumerate(curve.control_points):
            dwg.add(dwg.circle(center=(float(cp[0]), float(cp[1])), r=3, fill='red', opacity=0.7))
    
    dwg.save()
    print(f"SVG saved as {filename}")

def export_to_svg(data, filename="output.svg"):
    """Legacy function for backward compatibility"""
    print(f"Exporting data to {filename}...")
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for line in data:
        dwg.add(dwg.line(start=line[0], end=line[1], stroke=svgwrite.rgb(10, 10, 16, '%')))
    dwg.save()

def evaluate_quality(data):
    """Evaluate the quality of generated Bézier curves"""
    print("Evaluating quality of generated data...")
    
    if isinstance(data, np.ndarray):
        # Convert to list of curves if needed
        if data.ndim == 3:  # (num_curves, num_control_points, 2)
            curves = [BezierCurve(curve) for curve in data]
        else:
            return "Quality: Unable to evaluate - invalid data format"
    else:
        curves = data
    
    # Basic quality metrics
    quality_metrics = {
        'num_curves': len(curves),
        'avg_curve_length': 0.0,
        'control_point_spread': 0.0,
        'smoothness_score': 0.0
    }
    
    total_length = 0
    total_spread = 0
    
    for curve in curves:
        # Calculate approximate curve length
        points = curve.sample_points(50)
        lengths = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        curve_length = np.sum(lengths)
        total_length += curve_length
        
        # Calculate control point spread (measure of curve complexity)
        cp_distances = np.sqrt(np.sum(np.diff(curve.control_points, axis=0)**2, axis=1))
        spread = np.mean(cp_distances)
        total_spread += spread
    
    if len(curves) > 0:
        quality_metrics['avg_curve_length'] = total_length / len(curves)
        quality_metrics['control_point_spread'] = total_spread / len(curves)
    
    return f"Quality Metrics: {quality_metrics}"

def evaluate_diversity(curves):
    """Evaluate the diversity of generated Bézier curves"""
    print("Evaluating diversity of generated curves...")
    
    if len(curves) < 2:
        return "Diversity: Not enough curves to evaluate"
    
    # Calculate pairwise distances between curves
    distances = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            curve1 = curves[i] if isinstance(curves[i], BezierCurve) else BezierCurve(curves[i])
            curve2 = curves[j] if isinstance(curves[j], BezierCurve) else BezierCurve(curves[j])
            
            # Sample points and calculate distance
            points1 = curve1.sample_points(20)
            points2 = curve2.sample_points(20)
            
            # Calculate average distance between corresponding points
            distance = np.mean(np.sqrt(np.sum((points1 - points2)**2, axis=1)))
            distances.append(distance)
    
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    return f"Diversity: Avg distance={avg_distance:.2f}, Std={std_distance:.2f}"
