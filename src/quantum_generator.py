"""
quantum_generator.py
This file contains the implementation of a quantum-enhanced generator using PennyLane with GPU acceleration.
"""

import pennylane as qml
import numpy as np
import torch

# Try to import CUDA utilities
try:
    from cuda_utils import get_cuda_manager
    CUDA_UTILS_AVAILABLE = True
except ImportError:
    CUDA_UTILS_AVAILABLE = False

class QuantumGenerator:
    def __init__(self, num_qubits=4, use_gpu=True):
        self.num_qubits = num_qubits
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Setup device with GPU support if available
        if self.use_gpu:
            try:
                # Try to use lightning.gpu for faster quantum simulations
                self.dev = qml.device("lightning.gpu", wires=self.num_qubits)
                print("Using lightning.gpu device for quantum computations")
            except:
                try:
                    # Fallback to default.qubit with CUDA support
                    self.dev = qml.device("default.qubit", wires=self.num_qubits)
                    print("Using default.qubit device")
                except:
                    self.dev = qml.device("default.qubit", wires=self.num_qubits)
                    print("Using default.qubit device (CPU)")
        else:
            self.dev = qml.device("default.qubit", wires=self.num_qubits)
            print("Using default.qubit device (CPU)")
        
        # Initialize CUDA manager if available
        if CUDA_UTILS_AVAILABLE:
            self.cuda_manager = get_cuda_manager()
        else:
            self.cuda_manager = None

    def quantum_circuit(self, params):
        """Enhanced quantum circuit with more complex gates."""
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(params):
            # Layer 1: Rotation gates
            for i in range(self.num_qubits):
                qml.RX(params[i], wires=i)
                qml.RY(params[i + self.num_qubits], wires=i)
                qml.RZ(params[i + 2 * self.num_qubits], wires=i)
            
            # Layer 2: Entangling gates
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Layer 3: More rotations
            for i in range(self.num_qubits):
                qml.RY(params[i + 3 * self.num_qubits], wires=i)
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        return circuit

    def generate_bezier_curves(self, num_curves=10, num_control_points=4):
        """Generate Bézier curves using quantum circuits."""
        print(f"Generating {num_curves} quantum Bézier curves with {num_control_points} control points...")
        
        curves = []
        params_per_curve = 4 * self.num_qubits  # 4 layers of parameters
        
        for i in range(num_curves):
            # Generate random parameters for this curve
            params = np.random.uniform(0, 2 * np.pi, params_per_curve)
            
            # Convert to torch tensor if using GPU
            if self.use_gpu:
                params = torch.tensor(params, dtype=torch.float32)
                if torch.cuda.is_available():
                    params = params.cuda()
            
            # Execute quantum circuit
            circuit = self.quantum_circuit(params)
            quantum_output = circuit(params)
            
            # Convert quantum output to control points
            if isinstance(quantum_output, torch.Tensor):
                quantum_values = quantum_output.detach().cpu().numpy()
            else:
                quantum_values = np.array(quantum_output)
            
            # Map quantum values [-1, 1] to control points
            control_points = self._quantum_to_bezier_points(quantum_values, num_control_points)
            curves.append(control_points)
        
        return np.array(curves)

    def _quantum_to_bezier_points(self, quantum_values, num_control_points):
        """Convert quantum measurement values to Bézier control points."""
        # Expand quantum values to match required control points
        expanded_values = np.tile(quantum_values, (num_control_points * 2) // len(quantum_values) + 1)
        expanded_values = expanded_values[:num_control_points * 2]
        
        # Reshape to (num_control_points, 2) and scale to [0, 100]
        control_points = expanded_values.reshape(num_control_points, 2)
        control_points = (control_points + 1) * 50  # Scale from [-1, 1] to [0, 100]
        
        return control_points

    def hybrid_quantum_classical_generation(self, classical_generator, num_curves=10):
        """Combine quantum and classical generation for enhanced results."""
        print("Generating hybrid quantum-classical Bézier curves...")
        
        # Generate quantum-inspired latent vectors
        quantum_latents = []
        
        for _ in range(num_curves):
            # Generate quantum parameters
            params = np.random.uniform(0, 2 * np.pi, 4 * self.num_qubits)
            
            if self.use_gpu:
                params = torch.tensor(params, dtype=torch.float32)
                if torch.cuda.is_available():
                    params = params.cuda()
            
            # Execute quantum circuit
            circuit = self.quantum_circuit(params)
            quantum_output = circuit(params)
            
            if isinstance(quantum_output, torch.Tensor):
                quantum_values = quantum_output.detach().cpu().numpy()
            else:
                quantum_values = np.array(quantum_output)
            
            # Expand to latent dimension
            latent_dim = classical_generator.latent_dim
            expanded_latent = np.tile(quantum_values, latent_dim // len(quantum_values) + 1)[:latent_dim]
            quantum_latents.append(expanded_latent)
        
        # Use quantum latents with classical generator
        quantum_latents = np.array(quantum_latents)
        
        # Convert to torch tensor and move to appropriate device
        z = torch.FloatTensor(quantum_latents).to(classical_generator.device)
        
        # Generate curves using classical generator with quantum latents
        classical_generator.generator.eval()
        with torch.no_grad():
            generated_curves = classical_generator.generator(z)
            generated_curves = (generated_curves + 1) * 50  # Denormalize
        
        return generated_curves.cpu().numpy()

    def benchmark_quantum_performance(self, num_trials=100):
        """Benchmark quantum circuit performance."""
        print(f"Benchmarking quantum performance with {num_trials} trials...")
        
        import time
        
        params = np.random.uniform(0, 2 * np.pi, 4 * self.num_qubits)
        if self.use_gpu:
            params = torch.tensor(params, dtype=torch.float32)
            if torch.cuda.is_available():
                params = params.cuda()
        
        circuit = self.quantum_circuit(params)
        
        # Warm-up
        for _ in range(10):
            circuit(params)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_trials):
            result = circuit(params)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_trials
        print(f"Average quantum circuit execution time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {1/avg_time:.2f} circuits/second")
        
        return avg_time

    def generate(self, num_curves=5):
        """Main generation method for backward compatibility."""
        print("Generating with Enhanced Quantum Generator...")
        
        # Generate curves using quantum method
        curves = self.generate_bezier_curves(num_curves=num_curves, num_control_points=4)
        
        print(f"Generated {len(curves)} quantum Bézier curves")
        print("Sample curve (first control points):", curves[0][:2] if len(curves) > 0 else "None")
        
        # Export to SVG
        try:
            from utility import export_bezier_curves_to_svg
            export_bezier_curves_to_svg(curves, filename="quantum_bezier_curves.svg")
            print("Exported curves to quantum_bezier_curves.svg")
        except ImportError:
            print("Could not import export function")
        
        return curves
