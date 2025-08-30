"""
evaluation_metrics.py
Comprehensive evaluation metrics for quantum-generated geometric designs.
"""

import numpy as np
import torch
import torch.nn as nn
import math
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import CUDA utilities
try:
    from cuda_utils import get_cuda_manager, CUDABezierProcessor
    CUDA_UTILS_AVAILABLE = True
except ImportError:
    CUDA_UTILS_AVAILABLE = False

class FrechetInceptionDistance:
    """Fréchet Inception Distance (FID) for geometric designs."""
    
    def __init__(self, feature_extractor: Optional[nn.Module] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if feature_extractor is None:
            # Simple feature extractor for geometric data
            self.feature_extractor = self._create_geometric_feature_extractor()
        else:
            self.feature_extractor = feature_extractor
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
    
    def _create_geometric_feature_extractor(self) -> nn.Module:
        """Create a feature extractor for geometric designs."""
        return nn.Sequential(
            nn.Linear(8, 64),  # Assuming 4 control points * 2 coords
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # Feature dimension
        )
    
    def extract_features(self, geometries: torch.Tensor) -> torch.Tensor:
        """Extract features from geometric designs."""
        if geometries.dim() == 3:  # (batch, points, coords)
            geometries = geometries.view(geometries.size(0), -1)
        
        geometries = geometries.to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(geometries)
        
        return features.cpu()
    
    def calculate_fid(self, 
                     real_geometries: torch.Tensor, 
                     fake_geometries: torch.Tensor) -> float:
        """Calculate Fréchet Inception Distance."""
        # Extract features
        real_features = self.extract_features(real_geometries)
        fake_features = self.extract_features(fake_geometries)
        
        # Calculate statistics
        mu_real = torch.mean(real_features, dim=0)
        mu_fake = torch.mean(fake_features, dim=0)
        
        sigma_real = torch.cov(real_features.T)
        sigma_fake = torch.cov(fake_features.T)
        
        # Calculate FID
        diff = mu_real - mu_fake
        
        # Compute trace of covariance matrices
        trace_real = torch.trace(sigma_real)
        trace_fake = torch.trace(sigma_fake)
        
        # Compute product of covariances (simplified)
        try:
            # Full FID calculation with matrix square root
            sqrt_product = torch.linalg.matrix_power(
                sigma_real @ sigma_fake, 0.5
            )
            trace_sqrt = torch.trace(sqrt_product).real
        except:
            # Simplified version if matrix operations fail
            trace_sqrt = torch.sqrt(trace_real * trace_fake)
        
        fid = torch.sum(diff ** 2) + trace_real + trace_fake - 2 * trace_sqrt
        
        return float(fid)

class ModeCollapseDetector:
    """Detect mode collapse in generated geometric designs."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    def detect_mode_collapse(self, geometries: np.ndarray) -> Dict[str, float]:
        """Detect mode collapse using various metrics."""
        if len(geometries) < 2:
            return {"error": "Need at least 2 geometries"}
        
        # Flatten geometries for analysis
        flat_geometries = geometries.reshape(len(geometries), -1)
        
        # 1. Pairwise distance analysis
        distances = pdist(flat_geometries, metric='euclidean')
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        
        # 2. Clustering analysis
        n_clusters = min(len(geometries) // 2, 10)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(flat_geometries)
            silhouette = silhouette_score(flat_geometries, cluster_labels)
            
            # Count samples per cluster
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_balance = np.std(counts) / np.mean(counts)  # Lower is better
        else:
            silhouette = 0
            cluster_balance = 0
        
        # 3. Variance analysis
        feature_variances = np.var(flat_geometries, axis=0)
        mean_variance = np.mean(feature_variances)
        min_variance = np.min(feature_variances)
        
        # 4. Mode collapse indicators
        distance_collapse = mean_distance < self.threshold
        variance_collapse = mean_variance < self.threshold * 0.1
        
        return {
            'mean_pairwise_distance': float(mean_distance),
            'std_pairwise_distance': float(std_distance),
            'min_pairwise_distance': float(min_distance),
            'silhouette_score': float(silhouette),
            'cluster_balance': float(cluster_balance),
            'mean_feature_variance': float(mean_variance),
            'min_feature_variance': float(min_variance),
            'mode_collapse_detected': bool(distance_collapse or variance_collapse),
            'collapse_severity': float(max(0, 1 - mean_distance / self.threshold))
        }

class DiversityEvaluator:
    """Evaluate diversity of generated geometric designs."""
    
    def __init__(self):
        if CUDA_UTILS_AVAILABLE:
            self.cuda_manager = get_cuda_manager()
            self.processor = CUDABezierProcessor(self.cuda_manager)
        else:
            self.processor = None
    
    def calculate_diversity_metrics(self, geometries: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive diversity metrics."""
        if len(geometries) < 2:
            return {"error": "Need at least 2 geometries"}
        
        metrics = {}
        
        # 1. Geometric diversity
        geometric_metrics = self._calculate_geometric_diversity(geometries)
        metrics.update(geometric_metrics)
        
        # 2. Statistical diversity
        statistical_metrics = self._calculate_statistical_diversity(geometries)
        metrics.update(statistical_metrics)
        
        # 3. Topological diversity
        topological_metrics = self._calculate_topological_diversity(geometries)
        metrics.update(topological_metrics)
        
        # 4. Overall diversity score
        metrics['overall_diversity'] = self._compute_overall_diversity(metrics)
        
        return metrics
    
    def _calculate_geometric_diversity(self, geometries: np.ndarray) -> Dict[str, float]:
        """Calculate geometric diversity metrics."""
        metrics = {}
        
        # Curve lengths
        lengths = []
        for geometry in geometries:
            if self.processor:
                # Use CUDA acceleration
                batch = geometry[np.newaxis, ...]
                curve_metrics = self.processor.compute_curve_metrics(batch)
                lengths.append(curve_metrics['lengths'][0])
            else:
                # CPU calculation
                length = self._calculate_curve_length_cpu(geometry)
                lengths.append(length)
        
        lengths = np.array(lengths)
        metrics['length_diversity'] = float(np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0)
        metrics['length_range'] = float(np.max(lengths) - np.min(lengths))
        
        # Bounding box diversity
        bboxes = []
        for geometry in geometries:
            min_coords = np.min(geometry, axis=0)
            max_coords = np.max(geometry, axis=0)
            bbox_size = np.prod(max_coords - min_coords)
            bboxes.append(bbox_size)
        
        bboxes = np.array(bboxes)
        metrics['bbox_diversity'] = float(np.std(bboxes) / np.mean(bboxes) if np.mean(bboxes) > 0 else 0)
        
        # Aspect ratio diversity
        aspect_ratios = []
        for geometry in geometries:
            min_coords = np.min(geometry, axis=0)
            max_coords = np.max(geometry, axis=0)
            width = max_coords[0] - min_coords[0]
            height = max_coords[1] - min_coords[1]
            aspect_ratio = width / height if height > 0 else 1.0
            aspect_ratios.append(aspect_ratio)
        
        aspect_ratios = np.array(aspect_ratios)
        metrics['aspect_ratio_diversity'] = float(np.std(aspect_ratios))
        
        return metrics
    
    def _calculate_statistical_diversity(self, geometries: np.ndarray) -> Dict[str, float]:
        """Calculate statistical diversity metrics."""
        metrics = {}
        
        # Flatten geometries
        flat_geometries = geometries.reshape(len(geometries), -1)
        
        # Pairwise distances
        distances = pdist(flat_geometries, metric='euclidean')
        metrics['mean_pairwise_distance'] = float(np.mean(distances))
        metrics['std_pairwise_distance'] = float(np.std(distances))
        
        # Entropy-based diversity
        # Discretize the space and calculate entropy
        n_bins = min(20, len(geometries))
        entropies = []
        
        for dim in range(flat_geometries.shape[1]):
            values = flat_geometries[:, dim]
            hist, _ = np.histogram(values, bins=n_bins)
            # Add small epsilon to avoid log(0)
            hist_normalized = hist / np.sum(hist) + 1e-10
            ent = entropy(hist_normalized)
            entropies.append(ent)
        
        metrics['mean_entropy'] = float(np.mean(entropies))
        metrics['std_entropy'] = float(np.std(entropies))
        
        # Coverage diversity (how well the geometries cover the space)
        try:
            pca = PCA(n_components=min(2, flat_geometries.shape[1]))
            pca_features = pca.fit_transform(flat_geometries)
            
            # Calculate convex hull area (2D projection)
            if pca_features.shape[1] >= 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(pca_features)
                    metrics['coverage_area'] = float(hull.volume)  # 2D area
                except:
                    metrics['coverage_area'] = 0.0
            else:
                metrics['coverage_area'] = float(np.std(pca_features[:, 0]))
        except:
            metrics['coverage_area'] = 0.0
        
        return metrics
    
    def _calculate_topological_diversity(self, geometries: np.ndarray) -> Dict[str, float]:
        """Calculate topological diversity metrics."""
        metrics = {}
        
        # Curvature diversity
        curvatures = []
        for geometry in geometries:
            curvature = self._calculate_curvature_cpu(geometry)
            curvatures.extend(curvature)
        
        if curvatures:
            curvatures = np.array(curvatures)
            metrics['curvature_diversity'] = float(np.std(curvatures))
            metrics['max_curvature'] = float(np.max(curvatures))
            metrics['mean_curvature'] = float(np.mean(curvatures))
        else:
            metrics['curvature_diversity'] = 0.0
            metrics['max_curvature'] = 0.0
            metrics['mean_curvature'] = 0.0
        
        # Complexity diversity (number of inflection points, etc.)
        complexities = []
        for geometry in geometries:
            complexity = self._calculate_complexity_cpu(geometry)
            complexities.append(complexity)
        
        complexities = np.array(complexities)
        metrics['complexity_diversity'] = float(np.std(complexities))
        metrics['mean_complexity'] = float(np.mean(complexities))
        
        return metrics
    
    def _calculate_curve_length_cpu(self, control_points: np.ndarray, n_segments: int = 100) -> float:
        """CPU fallback for curve length calculation."""
        t_values = np.linspace(0, 1, n_segments + 1)
        points = []
        n = len(control_points) - 1
        
        for t in t_values:
            point = np.zeros(2)
            for i in range(n + 1):
                binom = math.comb(n, i)
                bernstein = binom * (t ** i) * ((1 - t) ** (n - i))
                point += bernstein * control_points[i]
            points.append(point)
        
        points = np.array(points)
        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return np.sum(lengths)
    
    def _calculate_curvature_cpu(self, control_points: np.ndarray) -> np.ndarray:
        """CPU fallback for curvature calculation."""
        t_values = np.linspace(0.1, 0.9, 10)  # Avoid endpoints
        curvatures = []
        dt = 0.01
        
        for t in t_values:
            # Numerical derivatives
            p1 = self._evaluate_bezier_at_t(control_points, max(0, t - dt))
            p2 = self._evaluate_bezier_at_t(control_points, t)
            p3 = self._evaluate_bezier_at_t(control_points, min(1, t + dt))
            
            # First and second derivatives
            v = (p3 - p1) / (2 * dt)
            a = (p3 - 2 * p2 + p1) / (dt ** 2)
            
            # Curvature
            cross_product = v[0] * a[1] - v[1] * a[0]
            velocity_magnitude = np.linalg.norm(v)
            
            if velocity_magnitude > 1e-10:
                curvature = abs(cross_product) / (velocity_magnitude ** 3)
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _calculate_complexity_cpu(self, control_points: np.ndarray) -> float:
        """Calculate geometric complexity."""
        # Simple complexity measure based on control point arrangement
        
        # 1. Convex hull ratio
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(control_points)
            hull_area = hull.volume
            
            # Bounding box area
            min_coords = np.min(control_points, axis=0)
            max_coords = np.max(control_points, axis=0)
            bbox_area = np.prod(max_coords - min_coords)
            
            convexity = hull_area / bbox_area if bbox_area > 0 else 0
        except:
            convexity = 0.5
        
        # 2. Control point spread
        center = np.mean(control_points, axis=0)
        distances = np.linalg.norm(control_points - center, axis=1)
        spread = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
        
        # 3. Angular variation
        angles = []
        for i in range(len(control_points) - 1):
            vec = control_points[i + 1] - control_points[i]
            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)
        
        angular_variation = np.std(angles) if len(angles) > 1 else 0
        
        # Combine metrics
        complexity = (1 - convexity) + spread + angular_variation / np.pi
        
        return complexity
    
    def _evaluate_bezier_at_t(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """Evaluate Bézier curve at parameter t."""
        n = len(control_points) - 1
        point = np.zeros(2)
        
        for i in range(n + 1):
            binom = math.comb(n, i)
            bernstein = binom * (t ** i) * ((1 - t) ** (n - i))
            point += bernstein * control_points[i]
        
        return point
    
    def _compute_overall_diversity(self, metrics: Dict[str, float]) -> float:
        """Compute overall diversity score."""
        key_metrics = [
            'length_diversity',
            'bbox_diversity', 
            'aspect_ratio_diversity',
            'mean_pairwise_distance',
            'mean_entropy',
            'curvature_diversity',
            'complexity_diversity'
        ]
        
        values = []
        for key in key_metrics:
            if key in metrics and not np.isnan(metrics[key]):
                values.append(metrics[key])
        
        if values:
            # Normalize and combine
            normalized_values = np.array(values) / (np.max(values) + 1e-10)
            return float(np.mean(normalized_values))
        else:
            return 0.0

class QualityEvaluator:
    """Evaluate quality of generated geometric designs."""
    
    def evaluate_quality(self, geometries: np.ndarray) -> Dict[str, float]:
        """Evaluate overall quality of geometric designs."""
        metrics = {}
        
        # 1. Smoothness evaluation
        smoothness_scores = []
        for geometry in geometries:
            smoothness = self._calculate_smoothness(geometry)
            smoothness_scores.append(smoothness)
        
        metrics['mean_smoothness'] = float(np.mean(smoothness_scores))
        metrics['std_smoothness'] = float(np.std(smoothness_scores))
        
        # 2. Geometric validity
        validity_scores = []
        for geometry in geometries:
            validity = self._check_geometric_validity(geometry)
            validity_scores.append(validity)
        
        metrics['validity_rate'] = float(np.mean(validity_scores))
        
        # 3. Aesthetic quality (simplified)
        aesthetic_scores = []
        for geometry in geometries:
            aesthetic = self._calculate_aesthetic_score(geometry)
            aesthetic_scores.append(aesthetic)
        
        metrics['mean_aesthetic'] = float(np.mean(aesthetic_scores))
        metrics['std_aesthetic'] = float(np.std(aesthetic_scores))
        
        # 4. Overall quality score
        metrics['overall_quality'] = (
            metrics['mean_smoothness'] * 0.3 +
            metrics['validity_rate'] * 0.4 +
            metrics['mean_aesthetic'] * 0.3
        )
        
        return metrics
    
    def _calculate_smoothness(self, control_points: np.ndarray) -> float:
        """Calculate smoothness of a curve."""
        # Calculate second derivatives (acceleration) at sample points
        t_values = np.linspace(0.1, 0.9, 10)
        dt = 0.01
        
        accelerations = []
        for t in t_values:
            # Numerical second derivative
            p1 = self._evaluate_bezier_at_t(control_points, max(0, t - dt))
            p2 = self._evaluate_bezier_at_t(control_points, t)
            p3 = self._evaluate_bezier_at_t(control_points, min(1, t + dt))
            
            acceleration = (p3 - 2 * p2 + p1) / (dt ** 2)
            acc_magnitude = np.linalg.norm(acceleration)
            accelerations.append(acc_magnitude)
        
        # Smoothness is inverse of acceleration variation
        if len(accelerations) > 1:
            smoothness = 1.0 / (1.0 + np.std(accelerations))
        else:
            smoothness = 1.0
        
        return smoothness
    
    def _check_geometric_validity(self, control_points: np.ndarray) -> float:
        """Check geometric validity of a curve."""
        validity_score = 1.0
        
        # 1. Check for NaN or inf values
        if np.any(np.isnan(control_points)) or np.any(np.isinf(control_points)):
            return 0.0
        
        # 2. Check reasonable coordinate ranges
        if np.any(control_points < -1000) or np.any(control_points > 1000):
            validity_score *= 0.5
        
        # 3. Check for degenerate cases (all points the same)
        if np.allclose(control_points, control_points[0]):
            return 0.0
        
        # 4. Check for reasonable spacing
        distances = []
        for i in range(len(control_points) - 1):
            dist = np.linalg.norm(control_points[i + 1] - control_points[i])
            distances.append(dist)
        
        if all(d < 1e-6 for d in distances):  # All points too close
            validity_score *= 0.3
        
        return validity_score
    
    def _calculate_aesthetic_score(self, control_points: np.ndarray) -> float:
        """Calculate aesthetic quality score."""
        # Simplified aesthetic evaluation based on geometric principles
        
        # 1. Golden ratio approximation
        bbox_min = np.min(control_points, axis=0)
        bbox_max = np.max(control_points, axis=0)
        width = bbox_max[0] - bbox_min[0]
        height = bbox_max[1] - bbox_min[1]
        
        if height > 0:
            aspect_ratio = width / height
            golden_ratio = 1.618
            golden_score = 1.0 / (1.0 + abs(aspect_ratio - golden_ratio))
        else:
            golden_score = 0.5
        
        # 2. Symmetry evaluation
        center = np.mean(control_points, axis=0)
        symmetry_score = 1.0 - np.std([
            np.linalg.norm(point - center) for point in control_points
        ]) / (np.linalg.norm(bbox_max - bbox_min) + 1e-10)
        
        # 3. Simplicity (prefer fewer sharp turns)
        curvature_evaluator = DiversityEvaluator()
        curvatures = curvature_evaluator._calculate_curvature_cpu(control_points)
        max_curvature = np.max(curvatures) if len(curvatures) > 0 else 0
        simplicity_score = 1.0 / (1.0 + max_curvature)
        
        # Combine scores
        aesthetic_score = (golden_score + symmetry_score + simplicity_score) / 3.0
        
        return aesthetic_score
    
    def _evaluate_bezier_at_t(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """Evaluate Bézier curve at parameter t."""
        n = len(control_points) - 1
        point = np.zeros(2)
        
        for i in range(n + 1):
            binom = math.comb(n, i)
            bernstein = binom * (t ** i) * ((1 - t) ** (n - i))
            point += bernstein * control_points[i]
        
        return point

class ComprehensiveEvaluator:
    """Comprehensive evaluation system for quantum-generated geometries."""
    
    def __init__(self):
        self.fid_calculator = FrechetInceptionDistance()
        self.mode_detector = ModeCollapseDetector()
        self.diversity_evaluator = DiversityEvaluator()
        self.quality_evaluator = QualityEvaluator()
    
    def evaluate_generation_quality(self, 
                                  real_geometries: np.ndarray,
                                  generated_geometries: np.ndarray) -> Dict[str, Union[float, Dict]]:
        """Comprehensive evaluation of generated geometries."""
        
        results = {
            'timestamp': str(np.datetime64('now')),
            'n_real_samples': len(real_geometries),
            'n_generated_samples': len(generated_geometries)
        }
        
        # Convert to tensors for FID calculation
        real_tensor = torch.from_numpy(real_geometries).float()
        generated_tensor = torch.from_numpy(generated_geometries).float()
        
        # 1. Fréchet Inception Distance
        try:
            fid_score = self.fid_calculator.calculate_fid(real_tensor, generated_tensor)
            results['fid_score'] = fid_score
        except Exception as e:
            results['fid_score'] = None
            results['fid_error'] = str(e)
        
        # 2. Mode collapse detection
        mode_collapse_results = self.mode_detector.detect_mode_collapse(generated_geometries)
        results['mode_collapse'] = mode_collapse_results
        
        # 3. Diversity evaluation
        diversity_results = self.diversity_evaluator.calculate_diversity_metrics(generated_geometries)
        results['diversity'] = diversity_results
        
        # 4. Quality evaluation
        quality_results = self.quality_evaluator.evaluate_quality(generated_geometries)
        results['quality'] = quality_results
        
        # 5. Comparative analysis
        if len(real_geometries) > 0:
            real_diversity = self.diversity_evaluator.calculate_diversity_metrics(real_geometries)
            real_quality = self.quality_evaluator.evaluate_quality(real_geometries)
            
            results['comparative_analysis'] = {
                'diversity_ratio': (
                    diversity_results.get('overall_diversity', 0) / 
                    real_diversity.get('overall_diversity', 1)
                ),
                'quality_ratio': (
                    quality_results.get('overall_quality', 0) / 
                    real_quality.get('overall_quality', 1)
                )
            }
        
        # 6. Overall score
        results['overall_score'] = self._calculate_overall_score(results)
        
        return results
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall generation quality score."""
        scores = []
        
        # FID score (lower is better, normalize)
        if results.get('fid_score') is not None:
            fid_normalized = 1.0 / (1.0 + results['fid_score'] / 100.0)
            scores.append(fid_normalized)
        
        # Diversity score
        diversity = results.get('diversity', {})
        if 'overall_diversity' in diversity:
            scores.append(diversity['overall_diversity'])
        
        # Quality score
        quality = results.get('quality', {})
        if 'overall_quality' in quality:
            scores.append(quality['overall_quality'])
        
        # Mode collapse penalty
        mode_collapse = results.get('mode_collapse', {})
        if not mode_collapse.get('mode_collapse_detected', True):
            scores.append(1.0)
        else:
            collapse_severity = mode_collapse.get('collapse_severity', 1.0)
            scores.append(1.0 - collapse_severity)
        
        if scores:
            return float(np.mean(scores))
        else:
            return 0.0
    
    def generate_evaluation_report(self, 
                                 real_geometries: np.ndarray,
                                 generated_geometries: np.ndarray,
                                 filename: str = "evaluation_report.json") -> str:
        """Generate a comprehensive evaluation report."""
        
        results = self.evaluate_generation_quality(real_geometries, generated_geometries)
        
        # Add human-readable summary
        results['summary'] = self._generate_summary(results)
        
        # Save to file
        with open(filename, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        return filename
    
    def _generate_summary(self, results: Dict) -> Dict[str, str]:
        """Generate human-readable summary."""
        summary = {}
        
        # Overall assessment
        overall_score = results.get('overall_score', 0)
        if overall_score > 0.8:
            summary['overall_assessment'] = "Excellent"
        elif overall_score > 0.6:
            summary['overall_assessment'] = "Good"
        elif overall_score > 0.4:
            summary['overall_assessment'] = "Fair"
        else:
            summary['overall_assessment'] = "Poor"
        
        # FID assessment
        fid_score = results.get('fid_score')
        if fid_score is not None:
            if fid_score < 50:
                summary['fid_assessment'] = "Low FID - High similarity to real data"
            elif fid_score < 200:
                summary['fid_assessment'] = "Moderate FID - Reasonable similarity"
            else:
                summary['fid_assessment'] = "High FID - Low similarity to real data"
        
        # Diversity assessment
        diversity = results.get('diversity', {})
        diversity_score = diversity.get('overall_diversity', 0)
        if diversity_score > 0.7:
            summary['diversity_assessment'] = "High diversity"
        elif diversity_score > 0.4:
            summary['diversity_assessment'] = "Moderate diversity"
        else:
            summary['diversity_assessment'] = "Low diversity"
        
        # Mode collapse assessment
        mode_collapse = results.get('mode_collapse', {})
        if mode_collapse.get('mode_collapse_detected', False):
            severity = mode_collapse.get('collapse_severity', 0)
            if severity > 0.7:
                summary['mode_collapse_assessment'] = "Severe mode collapse detected"
            else:
                summary['mode_collapse_assessment'] = "Mild mode collapse detected"
        else:
            summary['mode_collapse_assessment'] = "No mode collapse detected"
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    import math
    
    # Generate sample data
    print("Testing evaluation metrics...")
    
    # Create sample real and generated geometries
    np.random.seed(42)
    
    # Real geometries (more structured)
    real_geometries = []
    for i in range(100):
        # Create structured Bézier curves
        t = i / 100.0 * 2 * np.pi
        center = np.array([50, 50])
        radius = 30 + 10 * np.sin(3 * t)
        
        control_points = np.array([
            center + radius * np.array([np.cos(t), np.sin(t)]),
            center + radius * np.array([np.cos(t + 0.5), np.sin(t + 0.5)]),
            center + radius * np.array([np.cos(t + 1.0), np.sin(t + 1.0)]),
            center + radius * np.array([np.cos(t + 1.5), np.sin(t + 1.5)])
        ])
        real_geometries.append(control_points)
    
    real_geometries = np.array(real_geometries)
    
    # Generated geometries (more random)
    generated_geometries = np.random.randn(80, 4, 2) * 20 + 50
    
    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_generation_quality(real_geometries, generated_geometries)
    
    print("Evaluation Results:")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"FID Score: {results.get('fid_score', 'N/A')}")
    print(f"Mode Collapse Detected: {results['mode_collapse']['mode_collapse_detected']}")
    print(f"Diversity Score: {results['diversity']['overall_diversity']:.3f}")
    print(f"Quality Score: {results['quality']['overall_quality']:.3f}")
    
    # Generate report
    report_file = evaluator.generate_evaluation_report(
        real_geometries, generated_geometries, "test_evaluation_report.json"
    )
    print(f"Detailed report saved to: {report_file}")
    
    print("Evaluation testing completed!")
