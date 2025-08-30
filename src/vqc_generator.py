"""
vqc_generator.py
Variational Quantum Circuit (VQC) generator for geometric design with CUDA support.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import json

# Import CUDA utilities
try:
    from cuda_utils import get_cuda_manager
    CUDA_UTILS_AVAILABLE = True
except ImportError:
    CUDA_UTILS_AVAILABLE = False

class QuantumDataEncoder:
    """Different quantum data encoding strategies."""
    
    @staticmethod
    def angle_encoding(data: np.ndarray, n_qubits: int) -> np.ndarray:
        """Encode classical data using angle encoding (rotation gates)."""
        # Normalize data to [0, 2π] range
        normalized_data = (data + 1) * np.pi  # Assuming input in [-1, 1]
        
        # Repeat/truncate to match qubit count
        if len(normalized_data) < n_qubits:
            encoded = np.tile(normalized_data, (n_qubits // len(normalized_data)) + 1)[:n_qubits]
        else:
            encoded = normalized_data[:n_qubits]
        
        return encoded
    
    @staticmethod
    def amplitude_encoding(data: np.ndarray, n_qubits: int) -> np.ndarray:
        """Encode classical data using amplitude encoding."""
        # Normalize to unit vector
        data_normalized = data / np.linalg.norm(data) if np.linalg.norm(data) > 0 else data
        
        # Pad or truncate to 2^n_qubits
        target_size = 2 ** n_qubits
        if len(data_normalized) < target_size:
            padded = np.zeros(target_size)
            padded[:len(data_normalized)] = data_normalized
            return padded
        else:
            return data_normalized[:target_size]
    
    @staticmethod
    def basis_encoding(data: np.ndarray, n_qubits: int) -> np.ndarray:
        """Encode classical data using basis encoding (computational basis)."""
        # Convert to binary representation
        binary_data = []
        for value in data:
            # Map [-1, 1] to [0, 1] then to binary
            normalized = (value + 1) / 2
            binary = bin(int(normalized * (2**n_qubits - 1)))[2:].zfill(n_qubits)
            binary_data.extend([int(b) for b in binary])
        
        return np.array(binary_data[:n_qubits])

class VariationalQuantumCircuit:
    """Variational Quantum Circuit for geometric parameter generation."""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, encoding_type: str = "angle"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        
        # Initialize device with GPU support if available
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            try:
                self.dev = qml.device("lightning.gpu", wires=n_qubits)
                print(f"VQC using lightning.gpu device with {n_qubits} qubits")
            except:
                self.dev = qml.device("default.qubit", wires=n_qubits)
                print(f"VQC using default.qubit device with {n_qubits} qubits")
        else:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize CUDA manager
        if CUDA_UTILS_AVAILABLE:
            self.cuda_manager = get_cuda_manager()
        
        # Calculate parameter dimensions
        self.n_params = self._calculate_param_count()
        
        # Create quantum circuit
        self.circuit = self._create_circuit()
    
    def _calculate_param_count(self) -> int:
        """Calculate total number of variational parameters."""
        # Each layer has rotation gates for each qubit (3 rotations: RX, RY, RZ)
        # Plus entangling gates (no additional parameters for CNOT)
        return self.n_layers * self.n_qubits * 3
    
    def _create_circuit(self):
        """Create the variational quantum circuit."""
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(params, x=None):
            # Data encoding layer (if input data provided)
            if x is not None:
                self._encode_data(x)
            
            # Variational layers
            param_idx = 0
            for layer in range(self.n_layers):
                # Rotation gates
                for qubit in range(self.n_qubits):
                    qml.RX(params[param_idx], wires=qubit)
                    qml.RY(params[param_idx + 1], wires=qubit)
                    qml.RZ(params[param_idx + 2], wires=qubit)
                    param_idx += 3
                
                # Entangling layer
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                
                # Additional entanglement for better expressivity
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def _encode_data(self, x):
        """Encode classical data into quantum state."""
        if self.encoding_type == "angle":
            encoded = QuantumDataEncoder.angle_encoding(x.detach().cpu().numpy(), self.n_qubits)
            for i, angle in enumerate(encoded):
                qml.RY(angle, wires=i)
        
        elif self.encoding_type == "amplitude":
            # Amplitude encoding requires state preparation
            encoded = QuantumDataEncoder.amplitude_encoding(x.detach().cpu().numpy(), self.n_qubits)
            qml.AmplitudeEmbedding(encoded, wires=range(self.n_qubits), normalize=True)
        
        elif self.encoding_type == "basis":
            encoded = QuantumDataEncoder.basis_encoding(x.detach().cpu().numpy(), self.n_qubits)
            for i, bit in enumerate(encoded):
                if bit == 1:
                    qml.PauliX(wires=i % self.n_qubits)
    
    def forward(self, params, x=None):
        """Forward pass through the quantum circuit."""
        if self.use_gpu and isinstance(params, torch.Tensor):
            if not params.is_cuda:
                params = params.cuda()
        
        return self.circuit(params, x)

class QGANGenerator(nn.Module):
    """Quantum-enhanced generator for geometric design parameters."""
    
    def __init__(self, 
                 latent_dim: int = 10,
                 n_qubits: int = 4,
                 n_layers: int = 3,
                 output_dim: int = 8,  # 4 control points * 2 coordinates
                 encoding_type: str = "angle",
                 classical_postprocessing: bool = True):
        super(QGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.classical_postprocessing = classical_postprocessing
        
        # Initialize VQC
        self.vqc = VariationalQuantumCircuit(n_qubits, n_layers, encoding_type)
        
        # Variational parameters (learnable)
        self.quantum_params = nn.Parameter(
            torch.randn(self.vqc.n_params, requires_grad=True) * 0.1
        )
        
        # Classical preprocessing layer
        self.classical_input = nn.Sequential(
            nn.Linear(latent_dim, n_qubits),
            nn.Tanh()
        )
        
        # Classical postprocessing (if enabled)
        if classical_postprocessing:
            self.classical_output = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.Tanh()  # Output in [-1, 1]
            )
        else:
            # Direct mapping from quantum output
            self.classical_output = nn.Linear(n_qubits, output_dim)
    
    def forward(self, z):
        """Generate geometric parameters from latent vector."""
        batch_size = z.size(0)
        
        # Classical preprocessing
        x_processed = self.classical_input(z)
        
        # Quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            # Execute quantum circuit for each sample
            quantum_out = self.vqc.forward(self.quantum_params, x_processed[i])
            if isinstance(quantum_out, (list, tuple)):
                quantum_out = torch.stack([torch.tensor(x, dtype=torch.float32) for x in quantum_out])
            quantum_outputs.append(quantum_out)
        
        quantum_batch = torch.stack(quantum_outputs)
        
        # Move to appropriate device
        if z.is_cuda:
            quantum_batch = quantum_batch.cuda()
        
        # Classical postprocessing
        output = self.classical_output(quantum_batch)
        
        return output

class GeometryPostProcessor:
    """Post-process quantum generator output into geometric representations."""
    
    @staticmethod
    def to_control_points(params: torch.Tensor, n_points: int = 4, scale: float = 100.0) -> torch.Tensor:
        """Convert parameters to Bézier control points."""
        batch_size = params.size(0)
        
        # Reshape to (batch_size, n_points, 2)
        if params.size(1) != n_points * 2:
            # Interpolate or pad to correct size
            target_size = n_points * 2
            if params.size(1) < target_size:
                # Pad with zeros
                padding = torch.zeros(batch_size, target_size - params.size(1), device=params.device)
                params = torch.cat([params, padding], dim=1)
            else:
                # Truncate
                params = params[:, :target_size]
        
        control_points = params.view(batch_size, n_points, 2)
        
        # Scale from [-1, 1] to [0, scale]
        control_points = (control_points + 1) * scale / 2
        
        return control_points
    
    @staticmethod
    def to_curve_descriptors(params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert parameters to curve descriptors (center, radius, etc.)."""
        # Extract geometric descriptors
        center_x = params[:, 0:1] * 50 + 50  # Center X in [0, 100]
        center_y = params[:, 1:2] * 50 + 50  # Center Y in [0, 100]
        radius = torch.abs(params[:, 2:3]) * 30 + 10  # Radius in [10, 40]
        rotation = params[:, 3:4] * np.pi  # Rotation in [-π, π]
        
        # Additional shape parameters
        aspect_ratio = torch.abs(params[:, 4:5]) * 0.5 + 0.5  # Aspect ratio [0.5, 1.0]
        curvature = params[:, 5:6] if params.size(1) > 5 else torch.zeros_like(center_x)
        
        return {
            'center': torch.cat([center_x, center_y], dim=1),
            'radius': radius,
            'rotation': rotation,
            'aspect_ratio': aspect_ratio,
            'curvature': curvature
        }
    
    @staticmethod
    def to_contour_points(params: torch.Tensor, n_points: int = 16) -> torch.Tensor:
        """Convert parameters to 2D contour points."""
        batch_size = params.size(0)
        
        # Generate parametric contour
        t = torch.linspace(0, 2 * np.pi, n_points, device=params.device)
        
        contours = []
        for i in range(batch_size):
            # Extract parameters for this sample
            p = params[i]
            
            # Create parametric curve
            center_x = p[0] * 50 + 50
            center_y = p[1] * 50 + 50
            radius_x = torch.abs(p[2]) * 30 + 10
            radius_y = torch.abs(p[3]) * 30 + 10 if len(p) > 3 else radius_x
            
            # Generate elliptical contour
            x = center_x + radius_x * torch.cos(t)
            y = center_y + radius_y * torch.sin(t)
            
            # Add higher-order harmonics for complexity
            if len(p) > 4:
                for harmonic in range(2, min(5, len(p)//2)):
                    amplitude = p[harmonic*2] * 5
                    phase = p[harmonic*2 + 1] * np.pi
                    x += amplitude * torch.cos(harmonic * t + phase)
                    y += amplitude * torch.sin(harmonic * t + phase)
            
            contour = torch.stack([x, y], dim=1)  # Shape: (n_points, 2)
            contours.append(contour)
        
        return torch.stack(contours)  # Shape: (batch_size, n_points, 2)

class QuantumGeometryGAN:
    """Complete Quantum GAN system for geometric design generation."""
    
    def __init__(self,
                 latent_dim: int = 10,
                 n_qubits: int = 4,
                 n_layers: int = 3,
                 output_dim: int = 8,
                 discriminator_hidden: List[int] = [64, 32, 16],
                 lr_g: float = 0.001,
                 lr_d: float = 0.0005,
                 encoding_type: str = "angle"):
        
        # Initialize generator
        self.generator = QGANGenerator(
            latent_dim=latent_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            output_dim=output_dim,
            encoding_type=encoding_type
        )
        
        # Initialize classical discriminator
        self.discriminator = self._create_discriminator(output_dim, discriminator_hidden)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Post-processor
        self.post_processor = GeometryPostProcessor()
        
        # Training statistics
        self.training_stats = {
            'g_losses': [],
            'd_losses': [],
            'quantum_params_history': []
        }
    
    def _create_discriminator(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Create classical discriminator network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def train_step(self, real_data: torch.Tensor) -> Tuple[float, float]:
        """Single training step."""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_data.view(batch_size, -1))
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_data = self.generator(z)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_output = self.discriminator(fake_data.detach().view(batch_size, -1))
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        fake_output = self.discriminator(fake_data.view(batch_size, -1))
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_g.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_geometries(self, 
                          num_samples: int = 10, 
                          output_type: str = "control_points") -> torch.Tensor:
        """Generate geometric designs."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.latent_dim, device=self.device)
            raw_output = self.generator(z)
            
            if output_type == "control_points":
                return self.post_processor.to_control_points(raw_output)
            elif output_type == "contour_points":
                return self.post_processor.to_contour_points(raw_output)
            elif output_type == "curve_descriptors":
                return self.post_processor.to_curve_descriptors(raw_output)
            else:
                return raw_output
    
    def get_quantum_params(self) -> torch.Tensor:
        """Get current quantum parameters."""
        return self.generator.quantum_params.detach().clone()
    
    def analyze_quantum_evolution(self) -> Dict:
        """Analyze how quantum parameters evolved during training."""
        if not self.training_stats['quantum_params_history']:
            return {"error": "No training history available"}
        
        params_history = torch.stack(self.training_stats['quantum_params_history'])
        
        return {
            'param_evolution': params_history,
            'param_variance': torch.var(params_history, dim=0),
            'param_mean': torch.mean(params_history, dim=0),
            'param_range': torch.max(params_history, dim=0)[0] - torch.min(params_history, dim=0)[0]
        }

def create_synthetic_geometry_dataset(num_samples: int = 1000, 
                                    geometry_type: str = "bezier") -> torch.Tensor:
    """Create synthetic geometry dataset for training."""
    
    if geometry_type == "bezier":
        # Generate random Bézier control points
        data = []
        for _ in range(num_samples):
            # Create meaningful Bézier curves
            start_point = torch.rand(2) * 100
            end_point = torch.rand(2) * 100
            
            # Control points that create smooth curves
            control1 = start_point + torch.randn(2) * 20
            control2 = end_point + torch.randn(2) * 20
            
            curve = torch.stack([start_point, control1, control2, end_point])
            data.append(curve)
        
        return torch.stack(data)
    
    elif geometry_type == "contour":
        # Generate elliptical contours with variations
        data = []
        for _ in range(num_samples):
            center = torch.rand(2) * 80 + 10  # Center in [10, 90]
            radius_x = torch.rand(1) * 30 + 10  # Radius in [10, 40]
            radius_y = torch.rand(1) * 30 + 10
            
            # Parameters: [center_x, center_y, radius_x, radius_y, ...]
            params = torch.cat([center, radius_x, radius_y])
            
            # Add random harmonics
            for _ in range(4):
                params = torch.cat([params, torch.randn(2) * 0.1])
            
            data.append(params)
        
        return torch.stack(data)
    
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")

# Example usage and testing
if __name__ == "__main__":
    # Test VQC generator
    print("Testing Variational Quantum Circuit Generator...")
    
    # Create QGAN
    qgan = QuantumGeometryGAN(
        latent_dim=8,
        n_qubits=4,
        n_layers=2,
        output_dim=8
    )
    
    print(f"Generator device: {next(qgan.generator.parameters()).device}")
    print(f"Quantum parameters shape: {qgan.generator.quantum_params.shape}")
    
    # Generate sample geometries
    geometries = qgan.generate_geometries(num_samples=5, output_type="control_points")
    print(f"Generated geometries shape: {geometries.shape}")
    print("Sample geometry (control points):")
    print(geometries[0].cpu().numpy())
    
    # Test different encoding types
    for encoding in ["angle", "amplitude", "basis"]:
        print(f"\nTesting {encoding} encoding...")
        try:
            qgan_test = QuantumGeometryGAN(
                latent_dim=6,
                n_qubits=3,
                n_layers=2,
                output_dim=6,
                encoding_type=encoding
            )
            test_geometries = qgan_test.generate_geometries(num_samples=3)
            print(f"✓ {encoding} encoding successful, output shape: {test_geometries.shape}")
        except Exception as e:
            print(f"✗ {encoding} encoding failed: {e}")
