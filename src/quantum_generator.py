"""
quantum_generator.py
This file contains the implementation of a quantum-enhanced generator using PennyLane.
"""

import pennylane as qml
import numpy as np

class QuantumGenerator:
    def __init__(self):
        self.num_qubits = 4
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

    def circuit(self, params):
        @qml.qnode(self.dev)
        def quantum_circuit(params):
            for i in range(self.num_qubits):
                qml.RX(params[i], wires=i)
                qml.RY(params[i + self.num_qubits], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return quantum_circuit(params)

    def generate(self):
        print("Generating with Quantum Generator...")
        params = np.random.uniform(0, np.pi, 2 * self.num_qubits)
        output = self.circuit(params)
        print("Quantum Output:", output)

        # Example data for SVG export
        data = [((0, 0), (output[0] * 100, output[1] * 100))]
        from src.utility import export_to_svg
        export_to_svg(data, filename="quantum_output.svg")
