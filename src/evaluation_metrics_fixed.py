"""
evaluation_metrics_fixed.py
Fixed version of evaluation metrics for the demo.
"""

import numpy as np
import torch
import math
from typing import Dict, Union
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class SimpleEvaluator:
    """Simplified evaluator for demonstration purposes."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate_generation_quality(self, real_data: np.ndarray, generated_data: np.ndarray) -> Dict[str, Union[float, Dict]]:
        """Evaluate generation quality with simplified metrics."""
        try:
            results = {}
            
            # 1. Basic diversity evaluation
            diversity_results = self.evaluate_diversity(generated_data)
            results['diversity'] = diversity_results
            
            # 2. Simple quality metrics
            quality_results = self.evaluate_quality(generated_data)
            results['quality'] = quality_results
            
            # 3. Mode collapse detection
            mode_collapse_results = self.detect_mode_collapse(generated_data)
            results['mode_collapse'] = mode_collapse_results
            
            # 4. Overall score (simple average)
            overall_score = (
                diversity_results.get('overall_diversity', 0) * 0.4 +
                quality_results.get('overall_quality', 0) * 0.4 +
                (1.0 - mode_collapse_results.get('collapse_score', 0)) * 0.2
            )
            results['overall_score'] = float(overall_score)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'overall_score': 0.0,
                'diversity': {'overall_diversity': 0.0},
                'quality': {'overall_quality': 0.0},
                'mode_collapse': {'mode_collapse_detected': True, 'collapse_score': 1.0}
            }
    
    def evaluate_diversity(self, geometries: np.ndarray) -> Dict[str, float]:
        """Evaluate diversity of generated geometries."""
        if len(geometries) < 2:
            return {'overall_diversity': 0.0, 'pairwise_distance_mean': 0.0}
        
        # Flatten geometries
        flat_geometries = geometries.reshape(len(geometries), -1)
        
        # Calculate pairwise distances
        distances = pdist(flat_geometries, metric='euclidean')
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))
        
        # Normalize diversity score (0-1)
        diversity_score = min(1.0, mean_distance / 100.0)  # Simple normalization
        
        return {
            'overall_diversity': diversity_score,
            'pairwise_distance_mean': mean_distance,
            'pairwise_distance_std': std_distance,
            'distance_range': float(np.max(distances) - np.min(distances))
        }
    
    def evaluate_quality(self, geometries: np.ndarray) -> Dict[str, float]:
        """Evaluate quality of generated geometries."""
        if len(geometries) < 1:
            return {'overall_quality': 0.0}
        
        quality_scores = []
        
        for geometry in geometries:
            # Simple quality metrics
            
            # 1. Control point spread (higher is better)
            if geometry.ndim == 2:  # (points, coords)
                spread = float(np.std(geometry))
                spread_score = min(1.0, spread / 50.0)  # Normalize
            else:
                spread_score = 0.5
            
            # 2. Smoothness (lower variance in distances is better)
            if len(geometry) > 1:
                point_distances = [np.linalg.norm(geometry[i] - geometry[i-1]) 
                                 for i in range(1, len(geometry))]
                smoothness = 1.0 - min(1.0, np.std(point_distances) / 20.0)
            else:
                smoothness = 0.5
            
            # 3. Geometric validity (points within reasonable range)
            min_coord = np.min(geometry)
            max_coord = np.max(geometry)
            coord_range = max_coord - min_coord
            validity_score = 1.0 if 10 < coord_range < 200 else 0.5
            
            # Combine scores
            quality_score = (spread_score + smoothness + validity_score) / 3.0
            quality_scores.append(quality_score)
        
        overall_quality = float(np.mean(quality_scores))
        
        return {
            'overall_quality': overall_quality,
            'individual_scores': quality_scores,
            'quality_std': float(np.std(quality_scores))
        }
    
    def detect_mode_collapse(self, geometries: np.ndarray) -> Dict[str, Union[bool, float]]:
        """Detect mode collapse in generated geometries."""
        if len(geometries) < 3:
            return {'mode_collapse_detected': False, 'collapse_score': 0.0}
        
        # Flatten geometries
        flat_geometries = geometries.reshape(len(geometries), -1)
        
        # Calculate pairwise distances
        distances = pdist(flat_geometries, metric='euclidean')
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)
        
        # Simple mode collapse detection
        # If minimum distance is very small compared to mean, might be mode collapse
        if mean_distance > 0:
            collapse_ratio = min_distance / mean_distance
            mode_collapse_detected = collapse_ratio < 0.1  # Threshold
            collapse_score = float(1.0 - collapse_ratio)
        else:
            mode_collapse_detected = True
            collapse_score = 1.0
        
        return {
            'mode_collapse_detected': mode_collapse_detected,
            'collapse_score': collapse_score,
            'min_distance': float(min_distance),
            'mean_distance': float(mean_distance)
        }
    
    def generate_evaluation_report(self, real_data: np.ndarray, generated_data: np.ndarray, filename: str) -> str:
        """Generate a simple evaluation report."""
        results = self.evaluate_generation_quality(real_data, generated_data)
        
        report = {
            'evaluation_summary': {
                'real_samples': len(real_data),
                'generated_samples': len(generated_data),
                'overall_score': results.get('overall_score', 0.0)
            },
            'detailed_results': results,
            'interpretation': {
                'diversity': 'High' if results.get('diversity', {}).get('overall_diversity', 0) > 0.7 else 'Medium' if results.get('diversity', {}).get('overall_diversity', 0) > 0.4 else 'Low',
                'quality': 'High' if results.get('quality', {}).get('overall_quality', 0) > 0.7 else 'Medium' if results.get('quality', {}).get('overall_quality', 0) > 0.4 else 'Low',
                'mode_collapse': 'Detected' if results.get('mode_collapse', {}).get('mode_collapse_detected', False) else 'Not detected'
            }
        }
        
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            return filename
        except Exception as e:
            print(f"Failed to save report: {e}")
            return "report_failed.json"

# Create an alias for backward compatibility
ComprehensiveEvaluator = SimpleEvaluator
