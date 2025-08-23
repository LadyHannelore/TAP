"""
bezier_utils.py
Utility functions for working with Bézier curves
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

class BezierCurve:
    def __init__(self, control_points):
        """
        Initialize a Bézier curve with control points.
        control_points: array of shape (n, 2) where n is number of control points
        """
        self.control_points = np.array(control_points)
        
    def evaluate(self, t):
        """
        Evaluate the Bézier curve at parameter t (0 <= t <= 1)
        """
        n = len(self.control_points) - 1
        result = np.zeros(2)
        for i in range(n + 1):
            binomial_coeff = self._binomial_coefficient(n, i)
            result += binomial_coeff * (t ** i) * ((1 - t) ** (n - i)) * self.control_points[i]
        return result
    
    def _binomial_coefficient(self, n, k):
        """Calculate binomial coefficient (n choose k)"""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def sample_points(self, num_points=100):
        """Sample points along the Bézier curve"""
        t_values = np.linspace(0, 1, num_points)
        points = np.array([self.evaluate(t) for t in t_values])
        return points
    
    def to_svg_path(self):
        """Convert Bézier curve to SVG path string"""
        if len(self.control_points) == 4:  # Cubic Bézier
            cp = self.control_points
            return f"M {float(cp[0][0])},{float(cp[0][1])} C {float(cp[1][0])},{float(cp[1][1])} {float(cp[2][0])},{float(cp[2][1])} {float(cp[3][0])},{float(cp[3][1])}"
        elif len(self.control_points) == 3:  # Quadratic Bézier
            cp = self.control_points
            return f"M {float(cp[0][0])},{float(cp[0][1])} Q {float(cp[1][0])},{float(cp[1][1])} {float(cp[2][0])},{float(cp[2][1])}"
        else:
            # For higher order curves, sample points and create a polyline
            points = self.sample_points()
            path_str = f"M {float(points[0][0])},{float(points[0][1])}"
            for point in points[1:]:
                path_str += f" L {float(point[0])},{float(point[1])}"
            return path_str

def data_to_bezier_curves(data, num_control_points=4):
    """
    Convert raw data to Bézier curve control points.
    This function needs to be adapted based on your data format.
    """
    # Assuming data represents flattened control points
    # Reshape and normalize to reasonable coordinate range
    data_normalized = (data.astype(np.float32) + 128) / 255.0  # Normalize int8 to [0, 1]
    
    # Calculate how many curves we can extract
    points_per_curve = num_control_points * 2  # x, y for each control point
    num_curves = len(data_normalized) // points_per_curve
    
    curves = []
    for i in range(num_curves):
        start_idx = i * points_per_curve
        end_idx = start_idx + points_per_curve
        curve_data = data_normalized[start_idx:end_idx]
        
        # Reshape to control points (num_control_points, 2)
        control_points = curve_data.reshape(num_control_points, 2)
        # Scale to reasonable coordinate range (0-100)
        control_points *= 100
        
        curves.append(BezierCurve(control_points))
    
    return curves

def plot_bezier_curves(curves, title="Bézier Curves"):
    """Plot multiple Bézier curves"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, curve in enumerate(curves):
        # Plot the curve
        points = curve.sample_points()
        ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.7, label=f'Curve {i+1}' if i < 5 else "")
        
        # Plot control points
        ax.plot(curve.control_points[:, 0], curve.control_points[:, 1], 'ro', markersize=4)
        
        # Draw control polygon
        ax.plot(curve.control_points[:, 0], curve.control_points[:, 1], 'r--', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    if len(curves) <= 5:
        ax.legend()
    
    plt.tight_layout()
    return fig, ax
