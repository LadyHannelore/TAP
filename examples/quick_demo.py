"""
quick_demo.py
Quick demonstration of classical GAN with minimal epochs for fast testing.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def quick_classical_demo():
    """Run a quick demo with minimal training."""
    print("🚀 QUICK CLASSICAL GAN DEMO")
    print("Real-time training with 10 epochs for fast demonstration")
    print("=" * 60)
    
    # Import the classical GAN
    import classical_gan
    
    # Modify the training function to use fewer epochs
    gan, train_data = classical_gan.train_classical_gan(
        epochs=10,           # Quick demo with only 10 epochs
        batch_size=64,       # Larger batch size for faster training
        save_interval=5      # Save samples every 5 epochs
    )
    
    # Generate a few designs
    designs = classical_gan.generate_and_export_designs(gan, num_designs=5)
    
    print("\n🎉 QUICK DEMO COMPLETED!")
    print("Generated files:")
    print("  • classical_designs.json")  
    print("  • classical_designs.svg")
    print("  • training_progress.png")
    print("\nFor full training, run: python main.py")

if __name__ == "__main__":
    quick_classical_demo()
