"""
cad_export.py
CAD-friendly export functionality for geometric designs with CUDA acceleration.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import math
from dataclasses import dataclass
from enum import Enum

# Import CUDA utilities
try:
    from cuda_utils import CUDABezierProcessor, get_cuda_manager
    CUDA_UTILS_AVAILABLE = True
except ImportError:
    CUDA_UTILS_AVAILABLE = False

class ExportFormat(Enum):
    """Supported export formats."""
    SVG = "svg"
    BEZIER = "bezier"
    DXF = "dxf"
    JSON = "json"
    STEP = "step"  # Future implementation
    IGES = "iges"  # Future implementation

@dataclass
class GeometryMetadata:
    """Metadata for geometric designs."""
    name: str
    creation_time: str
    generator_type: str
    quantum_params: Optional[List[float]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    design_parameters: Optional[Dict[str, float]] = None

class BezierCurve:
    """Enhanced Bézier curve representation with CAD functionality."""
    
    def __init__(self, control_points: np.ndarray, metadata: Optional[GeometryMetadata] = None):
        self.control_points = np.array(control_points)
        self.metadata = metadata or GeometryMetadata("curve", "", "unknown")
        self.degree = len(control_points) - 1
        
        # CUDA acceleration if available
        if CUDA_UTILS_AVAILABLE:
            self.cuda_manager = get_cuda_manager()
            self.processor = CUDABezierProcessor(self.cuda_manager)
        else:
            self.cuda_manager = None
            self.processor = None
    
    def evaluate(self, t_values: np.ndarray) -> np.ndarray:
        """Evaluate curve at parameter values t."""
        if self.processor:
            # Use CUDA acceleration
            control_points_batch = self.control_points[np.newaxis, ...]  # Add batch dimension
            results = self.processor.batch_evaluate_curves(control_points_batch, len(t_values))
            return results[0]  # Remove batch dimension
        else:
            # CPU evaluation
            return self._cpu_evaluate(t_values)
    
    def _cpu_evaluate(self, t_values: np.ndarray) -> np.ndarray:
        """CPU fallback for curve evaluation."""
        n = self.degree
        points = []
        
        for t in t_values:
            point = np.zeros(2)
            for i in range(n + 1):
                # Binomial coefficient
                binom = math.comb(n, i)
                # Bernstein polynomial
                bernstein = binom * (t ** i) * ((1 - t) ** (n - i))
                point += bernstein * self.control_points[i]
            points.append(point)
        
        return np.array(points)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box of the curve."""
        # For precise bounding box, we should find extrema, but for simplicity:
        min_point = np.min(self.control_points, axis=0)
        max_point = np.max(self.control_points, axis=0)
        return min_point, max_point
    
    def get_length(self, num_segments: int = 100) -> float:
        """Approximate curve length."""
        t_values = np.linspace(0, 1, num_segments + 1)
        points = self.evaluate(t_values)
        
        # Calculate length as sum of segment lengths
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return np.sum(lengths)
    
    def get_curvature(self, t_values: np.ndarray) -> np.ndarray:
        """Calculate curvature at parameter values."""
        # First and second derivatives
        dt = 1e-6
        
        curvatures = []
        for t in t_values:
            # Numerical derivatives
            p1 = self.evaluate(np.array([max(0, t - dt)]))[0]
            p2 = self.evaluate(np.array([t]))[0]
            p3 = self.evaluate(np.array([min(1, t + dt)]))[0]
            
            # First derivative (velocity)
            v = (p3 - p1) / (2 * dt)
            # Second derivative (acceleration)
            a = (p3 - 2 * p2 + p1) / (dt ** 2)
            
            # Curvature formula: |v × a| / |v|^3
            cross_product = v[0] * a[1] - v[1] * a[0]  # 2D cross product
            velocity_magnitude = np.linalg.norm(v)
            
            if velocity_magnitude > 1e-10:
                curvature = abs(cross_product) / (velocity_magnitude ** 3)
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        return np.array(curvatures)

class SVGExporter:
    """Export geometric designs to SVG format."""
    
    def __init__(self, width: int = 800, height: int = 600, margin: int = 50):
        self.width = width
        self.height = height
        self.margin = margin
    
    def export_curves(self, 
                     curves: List[BezierCurve], 
                     filename: str,
                     include_control_points: bool = False,
                     include_metadata: bool = True) -> str:
        """Export Bézier curves to SVG."""
        
        # Create SVG root
        svg = ET.Element('svg', {
            'width': str(self.width),
            'height': str(self.height),
            'xmlns': 'http://www.w3.org/2000/svg',
            'viewBox': f'0 0 {self.width} {self.height}'
        })
        
        # Add styles
        style = ET.SubElement(svg, 'style')
        style.text = """
        .bezier-curve { fill: none; stroke: #2196F3; stroke-width: 2; }
        .control-point { fill: #FF5722; stroke: #D84315; stroke-width: 1; r: 3; }
        .control-line { stroke: #FFC107; stroke-width: 1; stroke-dasharray: 5,5; opacity: 0.7; }
        .metadata { font-family: Arial, sans-serif; font-size: 12px; fill: #333; }
        """
        
        # Calculate scaling to fit curves
        all_points = []
        for curve in curves:
            all_points.extend(curve.control_points)
        
        if all_points:
            all_points = np.array(all_points)
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            
            # Calculate scale and offset
            data_width = max_coords[0] - min_coords[0]
            data_height = max_coords[1] - min_coords[1]
            
            if data_width > 0 and data_height > 0:
                scale_x = (self.width - 2 * self.margin) / data_width
                scale_y = (self.height - 2 * self.margin) / data_height
                scale = min(scale_x, scale_y)
                
                offset_x = self.margin + (self.width - 2 * self.margin - data_width * scale) / 2
                offset_y = self.margin + (self.height - 2 * self.margin - data_height * scale) / 2
            else:
                scale = 1
                offset_x = offset_y = self.margin
        else:
            scale = 1
            offset_x = offset_y = self.margin
        
        def transform_point(p):
            return ((p[0] - min_coords[0]) * scale + offset_x,
                   (p[1] - min_coords[1]) * scale + offset_y)
        
        # Add curves
        for i, curve in enumerate(curves):
            # Transform control points
            transformed_points = [transform_point(p) for p in curve.control_points]
            
            # Create path element
            if len(transformed_points) == 4:  # Cubic Bézier
                path_data = f"M {transformed_points[0][0]},{transformed_points[0][1]} "
                path_data += f"C {transformed_points[1][0]},{transformed_points[1][1]} "
                path_data += f"{transformed_points[2][0]},{transformed_points[2][1]} "
                path_data += f"{transformed_points[3][0]},{transformed_points[3][1]}"
            else:
                # Quadratic or higher-order - approximate with cubic
                path_data = f"M {transformed_points[0][0]},{transformed_points[0][1]} "
                for j in range(1, len(transformed_points)):
                    path_data += f"L {transformed_points[j][0]},{transformed_points[j][1]} "
            
            path = ET.SubElement(svg, 'path', {
                'd': path_data,
                'class': 'bezier-curve',
                'id': f'curve_{i}'
            })
            
            # Add control points if requested
            if include_control_points:
                # Control lines
                for j in range(len(transformed_points) - 1):
                    line = ET.SubElement(svg, 'line', {
                        'x1': str(transformed_points[j][0]),
                        'y1': str(transformed_points[j][1]),
                        'x2': str(transformed_points[j + 1][0]),
                        'y2': str(transformed_points[j + 1][1]),
                        'class': 'control-line'
                    })
                
                # Control points
                for j, point in enumerate(transformed_points):
                    circle = ET.SubElement(svg, 'circle', {
                        'cx': str(point[0]),
                        'cy': str(point[1]),
                        'class': 'control-point',
                        'title': f'Control Point {j}'
                    })
        
        # Add metadata if requested
        if include_metadata and curves:
            metadata_group = ET.SubElement(svg, 'g', {'class': 'metadata'})
            y_offset = 20
            
            for i, curve in enumerate(curves):
                if curve.metadata:
                    text = ET.SubElement(metadata_group, 'text', {
                        'x': '10',
                        'y': str(y_offset),
                        'class': 'metadata'
                    })
                    text.text = f"Curve {i}: {curve.metadata.name} ({curve.metadata.generator_type})"
                    y_offset += 15
        
        # Write to file
        self._write_svg(svg, filename)
        return filename
    
    def _write_svg(self, svg_element: ET.Element, filename: str):
        """Write SVG element to file with proper formatting."""
        rough_string = ET.tostring(svg_element, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines
        lines = [line for line in pretty_string.split('\n') if line.strip()]
        pretty_string = '\n'.join(lines)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(pretty_string)

class DXFExporter:
    """Export geometric designs to DXF format (basic implementation)."""
    
    def export_curves(self, curves: List[BezierCurve], filename: str) -> str:
        """Export curves to DXF format."""
        dxf_content = self._create_dxf_header()
        
        # Add entities
        dxf_content += "0\nSECTION\n2\nENTITIES\n"
        
        for i, curve in enumerate(curves):
            # Convert Bézier to polyline approximation
            t_values = np.linspace(0, 1, 50)
            points = curve.evaluate(t_values)
            
            # Add polyline
            dxf_content += "0\nPOLYLINE\n"
            dxf_content += f"8\nLAYER_{i}\n"  # Layer name
            dxf_content += "70\n0\n"  # Polyline flag
            
            for point in points:
                dxf_content += "0\nVERTEX\n"
                dxf_content += f"10\n{point[0]:.6f}\n"  # X coordinate
                dxf_content += f"20\n{point[1]:.6f}\n"  # Y coordinate
                dxf_content += "30\n0.0\n"  # Z coordinate
            
            dxf_content += "0\nSEQEND\n"
        
        dxf_content += "0\nENDSEC\n"
        dxf_content += self._create_dxf_footer()
        
        with open(filename, 'w') as f:
            f.write(dxf_content)
        
        return filename
    
    def _create_dxf_header(self) -> str:
        """Create DXF header section."""
        return """0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
TABLES
0
TABLE
2
LAYER
70
1
0
LAYER
2
0
70
0
6
CONTINUOUS
0
ENDTAB
0
ENDSEC
"""
    
    def _create_dxf_footer(self) -> str:
        """Create DXF footer."""
        return "0\nEOF\n"

class JSONExporter:
    """Export geometric designs to JSON format."""
    
    def export_curves(self, curves: List[BezierCurve], filename: str) -> str:
        """Export curves to JSON format."""
        data = {
            'format_version': '1.0',
            'export_timestamp': str(np.datetime64('now')),
            'curves': []
        }
        
        for i, curve in enumerate(curves):
            try:
                # Safe length calculation
                try:
                    length = float(curve.get_length())
                except:
                    length = 0.0
                
                # Safe bounding box calculation
                try:
                    bbox_min, bbox_max = curve.get_bounding_box()
                    bounding_box = {
                        'min': bbox_min.tolist(),
                        'max': bbox_max.tolist()
                    }
                except:
                    bounding_box = {'min': [0, 0], 'max': [100, 100]}
                
                curve_data = {
                    'id': i,
                    'type': 'bezier',
                    'degree': curve.degree,
                    'control_points': curve.control_points.tolist(),
                    'metadata': {
                        'name': curve.metadata.name if curve.metadata else f'curve_{i}',
                        'generator_type': curve.metadata.generator_type if curve.metadata else 'unknown',
                        'length': length,
                        'bounding_box': bounding_box
                    }
                }
                
                if curve.metadata and curve.metadata.quantum_params:
                    curve_data['quantum_parameters'] = curve.metadata.quantum_params
                
                if curve.metadata and curve.metadata.quality_metrics:
                    curve_data['quality_metrics'] = curve.metadata.quality_metrics
                
                data['curves'].append(curve_data)
            
            except Exception as e:
                print(f"Warning: Failed to process curve {i}: {e}")
                # Add minimal curve data
                data['curves'].append({
                    'id': i,
                    'type': 'bezier',
                    'control_points': curve.control_points.tolist(),
                    'error': str(e)
                })
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            print(f"Failed to write JSON file: {e}")
            return None

class GeometryExporter:
    """Main exporter class for all supported formats."""
    
    def __init__(self):
        self.svg_exporter = SVGExporter()
        self.dxf_exporter = DXFExporter()
        self.json_exporter = JSONExporter()
    
    def export(self, 
               curves: Union[List[BezierCurve], np.ndarray, torch.Tensor],
               filename: str,
               format_type: ExportFormat,
               **kwargs) -> str:
        """Export curves to specified format."""
        
        # Convert input to BezierCurve objects if needed
        if isinstance(curves, (np.ndarray, torch.Tensor)):
            curves = self._convert_to_bezier_curves(curves)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Export based on format
        if format_type == ExportFormat.SVG:
            return self.svg_exporter.export_curves(curves, filename, **kwargs)
        
        elif format_type == ExportFormat.DXF:
            return self.dxf_exporter.export_curves(curves, filename)
        
        elif format_type == ExportFormat.JSON:
            return self.json_exporter.export_curves(curves, filename)
        
        elif format_type == ExportFormat.BEZIER:
            return self._export_bezier_native(curves, filename)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _convert_to_bezier_curves(self, data: Union[np.ndarray, torch.Tensor]) -> List[BezierCurve]:
        """Convert tensor data to BezierCurve objects."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        curves = []
        for i, control_points in enumerate(data):
            # Reshape if needed (assuming last dimension is coordinates)
            if control_points.ndim == 1:
                control_points = control_points.reshape(-1, 2)
            
            metadata = GeometryMetadata(
                name=f'generated_curve_{i}',
                creation_time=str(np.datetime64('now')),
                generator_type='quantum_gan'
            )
            
            curve = BezierCurve(control_points, metadata)
            curves.append(curve)
        
        return curves
    
    def _export_bezier_native(self, curves: List[BezierCurve], filename: str) -> str:
        """Export in native Bézier format (NumPy)."""
        data = []
        for curve in curves:
            data.append(curve.control_points)
        
        np.save(filename, np.array(data))
        return filename
    
    def batch_export(self,
                    curves: Union[List[BezierCurve], np.ndarray, torch.Tensor],
                    base_filename: str,
                    formats: List[ExportFormat],
                    **kwargs) -> Dict[ExportFormat, str]:
        """Export curves to multiple formats."""
        results = {}
        
        for format_type in formats:
            # Create filename with appropriate extension
            name_without_ext = os.path.splitext(base_filename)[0]
            filename = f"{name_without_ext}.{format_type.value}"
            
            try:
                exported_file = self.export(curves, filename, format_type, **kwargs)
                results[format_type] = exported_file
                print(f"✓ Exported to {format_type.value}: {exported_file}")
            except Exception as e:
                print(f"✗ Failed to export to {format_type.value}: {e}")
                results[format_type] = None
        
        return results

# CUDA-accelerated batch processing for large datasets
class CUDAGeometryProcessor:
    """CUDA-accelerated processing for large geometry datasets."""
    
    def __init__(self):
        if CUDA_UTILS_AVAILABLE:
            self.cuda_manager = get_cuda_manager()
            self.processor = CUDABezierProcessor(self.cuda_manager)
        else:
            self.cuda_manager = None
            self.processor = None
    
    def batch_evaluate_for_export(self, 
                                 control_points_batch: np.ndarray,
                                 resolution: int = 100) -> np.ndarray:
        """Batch evaluate curves for high-resolution export."""
        if self.processor:
            return self.processor.batch_evaluate_curves(control_points_batch, resolution)
        else:
            # CPU fallback
            results = []
            t_values = np.linspace(0, 1, resolution)
            
            for control_points in control_points_batch:
                curve = BezierCurve(control_points)
                points = curve.evaluate(t_values)
                results.append(points)
            
            return np.array(results)
    
    def compute_export_metrics(self, curves_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute metrics for export quality assessment."""
        if self.processor:
            return self.processor.compute_curve_metrics(curves_data)
        else:
            # CPU fallback
            metrics = {}
            curves = [BezierCurve(cp) for cp in curves_data]
            
            lengths = [curve.get_length() for curve in curves]
            metrics['lengths'] = np.array(lengths)
            metrics['mean_length'] = np.mean(lengths)
            metrics['std_length'] = np.std(lengths)
            
            return metrics

# Example usage and testing
if __name__ == "__main__":
    # Create sample curves
    curves = []
    
    # Sample Bézier curve 1
    control_points_1 = np.array([
        [10, 10],
        [30, 80],
        [70, 80],
        [90, 10]
    ])
    
    metadata_1 = GeometryMetadata(
        name="test_curve_1",
        creation_time=str(np.datetime64('now')),
        generator_type="quantum_vqc",
        quantum_params=[0.1, 0.5, 1.2, 0.8],
        quality_metrics={"smoothness": 0.85, "complexity": 0.6}
    )
    
    curve1 = BezierCurve(control_points_1, metadata_1)
    curves.append(curve1)
    
    # Sample Bézier curve 2
    control_points_2 = np.array([
        [20, 50],
        [40, 20],
        [60, 80],
        [80, 50]
    ])
    
    curve2 = BezierCurve(control_points_2)
    curves.append(curve2)
    
    # Test exports
    exporter = GeometryExporter()
    
    # Export to multiple formats
    formats = [ExportFormat.SVG, ExportFormat.JSON, ExportFormat.DXF]
    results = exporter.batch_export(curves, "test_output", formats, include_control_points=True)
    
    print("Export results:")
    for format_type, filename in results.items():
        if filename:
            print(f"✓ {format_type.value}: {filename}")
        else:
            print(f"✗ {format_type.value}: Failed")
    
    # Test CUDA processing
    if CUDA_UTILS_AVAILABLE:
        processor = CUDAGeometryProcessor()
        control_points_batch = np.array([control_points_1, control_points_2])
        
        # High-resolution evaluation
        high_res_curves = processor.batch_evaluate_for_export(control_points_batch, resolution=200)
        print(f"High-resolution curves shape: {high_res_curves.shape}")
        
        # Compute metrics
        metrics = processor.compute_export_metrics(control_points_batch)
        print("Curve metrics:", metrics)
    
    print("CAD export testing completed!")
