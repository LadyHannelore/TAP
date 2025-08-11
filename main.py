"""
main.py
This is the entry point for the Quantum GAN project.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.gan import ClassicalGAN
from src.quantum_generator import QuantumGenerator

def main():
    print("Starting Quantum GAN Project")

    # Initialize classical GAN
    gan = ClassicalGAN()
    gan.train()

    # Initialize quantum generator
    quantum_gen = QuantumGenerator()
    quantum_gen.generate()

if __name__ == "__main__":
    main()
