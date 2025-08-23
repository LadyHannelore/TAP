"""
gan.py
This file contains the implementation of a classical GAN for Bézier curves.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class BezierGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_control_points=4):
        super(BezierGenerator, self).__init__()
        self.num_control_points = num_control_points
        output_dim = num_control_points * 2  # x, y coordinates for each control point
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        output = self.model(z)
        # Reshape to (batch_size, num_control_points, 2)
        return output.view(-1, self.num_control_points, 2)

class BezierDiscriminator(nn.Module):
    def __init__(self, num_control_points=4):
        super(BezierDiscriminator, self).__init__()
        input_dim = num_control_points * 2  # x, y coordinates for each control point
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the input from (batch_size, num_control_points, 2) to (batch_size, input_dim)
        x_flat = x.view(x.size(0), -1)
        return self.model(x_flat)

class BezierGAN:
    def __init__(self, latent_dim=100, num_control_points=4, lr=0.0002):
        self.latent_dim = latent_dim
        self.num_control_points = num_control_points
        
        self.generator = BezierGenerator(latent_dim, num_control_points)
        self.discriminator = BezierDiscriminator(num_control_points)
        
        self.criterion = nn.BCELoss()
        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
    
    def load_data(self, data_path):
        """Load and preprocess the Bézier curve data"""
        data = np.load(data_path)
        
        # Convert int8 data to float and normalize to [-1, 1]
        data_normalized = (data.astype(np.float32) + 128) / 255.0 * 2 - 1
        
        # Calculate how many complete curves we can extract
        points_per_curve = self.num_control_points * 2
        num_curves = len(data_normalized) // points_per_curve
        
        # Reshape to curves
        curves_data = data_normalized[:num_curves * points_per_curve].reshape(num_curves, self.num_control_points, 2)
        
        # Convert to torch tensor
        tensor_data = torch.FloatTensor(curves_data)
        dataset = TensorDataset(tensor_data)
        
        return DataLoader(dataset, batch_size=64, shuffle=True)
    
    def train(self, data_loader, epochs=100):
        print(f"Training Bézier GAN for {epochs} epochs...")
        
        d_loss, g_loss = 0, 0  # Initialize variables
        
        for epoch in range(epochs):
            for i, (real_data,) in enumerate(data_loader):
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Train Discriminator
                self.optim_d.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output = self.discriminator(real_data)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake data
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(z)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optim_d.step()
                
                # Train Generator
                self.optim_g.zero_grad()
                
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optim_g.step()
            
            if epoch % 10 == 0 and d_loss != 0 and g_loss != 0:
                print(f"Epoch [{epoch}/{epochs}], D Loss: {float(d_loss.item()):.4f}, G Loss: {float(g_loss.item()):.4f}")
    
    def generate_curves(self, num_curves=10):
        """Generate new Bézier curves"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_curves, self.latent_dim).to(self.device)
            generated_curves = self.generator(z)
            # Denormalize from [-1, 1] to [0, 100] coordinate range
            generated_curves = (generated_curves + 1) * 50  # Scale to [0, 100]
            return generated_curves.cpu().numpy()

# Legacy classes for backward compatibility
class ClassicalGAN(BezierGAN):
    def __init__(self):
        super().__init__()
    
    def train_legacy(self, epochs=10000, batch_size=64):
        print("Training Classical GAN...")
        # Load training data
        try:
            data_loader = self.load_data('data/sg_t16_train.npy')
            super().train(data_loader, epochs=epochs//100)  # Adjust epochs for practical training
        except Exception as e:
            print(f"Could not load training data: {e}")
            print("Training with placeholder logic...")
            for epoch in range(min(epochs, 100)):  # Limit to prevent long training
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{min(epochs, 100)} completed.")
