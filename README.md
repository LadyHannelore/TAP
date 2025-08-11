# Quantum GANs for Synthetic Design Geometry Generation

## Project Description
This project explores the use of Quantum Generative Adversarial Networks (QGANs) to generate simple design geometries such as 2D contours, line curves, or part boundaries. The quantum-enhanced generator is combined with a classical discriminator to create a hybrid GAN model. The generated outputs are exported into CAD-friendly vector formats like SVG or Bézier representations.

## Features
- Classical GAN implementation.
- Quantum-enhanced generator using PennyLane.
- Export generated geometries to SVG files.
- Evaluate quality and diversity of generated outputs.

## Project Structure
```
TAP/
├── main.py                # Entry point for the project
├── src/
│   ├── gan.py            # Classical GAN implementation
│   ├── quantum_generator.py # Quantum generator implementation
│   ├── utility.py        # Utility functions (e.g., SVG export, evaluation)
└── README.md             # Project documentation
```

## Requirements
- Python 3.11
- Libraries: `pennylane`, `torch`, `torchvision`, `numpy`, `matplotlib`, `svgwrite`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LadyHannelore/TAP.git
   cd TAP
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```bash
   python main.py
   ```
2. Generated SVG files will be saved in the project directory.

## Future Work
- Enhance the quantum generator with more complex circuits.
- Add advanced evaluation metrics for quality and diversity.
- Extend support for Bézier curve representations.

## License
This project is licensed under the MIT License.

