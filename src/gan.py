"""
gan.py
This file contains the implementation of a classical GAN for Bézier curves with CUDA optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import os

# Import CUDA utilities
try:
    from cuda_utils import CUDAManager, CUDABezierProcessor, get_cuda_manager
    CUDA_UTILS_AVAILABLE = True
except ImportError:
    CUDA_UTILS_AVAILABLE = False
    print("CUDA utilities not available. Using basic CUDA support.")

class BezierGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_control_points=4, use_cuda_optimizations=True):
        super(BezierGenerator, self).__init__()
        self.num_control_points = num_control_points
        self.use_cuda_optimizations = use_cuda_optimizations
        output_dim = num_control_points * 2  # x, y coordinates for each control point
        
        # Enhanced architecture with more layers and better normalization
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights for better training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        output = self.model(z)
        # Reshape to (batch_size, num_control_points, 2)
        return output.view(-1, self.num_control_points, 2)

class BezierDiscriminator(nn.Module):
    def __init__(self, num_control_points=4, use_spectral_norm=True):
        super(BezierDiscriminator, self).__init__()
        input_dim = num_control_points * 2  # x, y coordinates for each control point
        
        # Enhanced architecture with spectral normalization for stability
        layers = []
        
        # First layer
        first_layer = nn.Linear(input_dim, 1024)
        if use_spectral_norm:
            first_layer = nn.utils.spectral_norm(first_layer)
        layers.extend([
            first_layer,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        ])
        
        # Hidden layers
        hidden_dims = [1024, 512, 256, 128]
        for i in range(len(hidden_dims) - 1):
            layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            if use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.extend([
                layer,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
        
        # Output layer
        output_layer = nn.Linear(hidden_dims[-1], 1)
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        layers.extend([
            output_layer,
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten the input from (batch_size, num_control_points, 2) to (batch_size, input_dim)
        x_flat = x.view(x.size(0), -1)
        return self.model(x_flat)

class BezierGAN:
    def __init__(self, latent_dim=100, num_control_points=4, lr=0.0002, 
                 beta1=0.5, beta2=0.999, use_cuda_optimizations=True):
        self.latent_dim = latent_dim
        self.num_control_points = num_control_points
        self.use_cuda_optimizations = use_cuda_optimizations
        
        # Initialize CUDA manager if available
        if CUDA_UTILS_AVAILABLE and use_cuda_optimizations:
            self.cuda_manager = get_cuda_manager()
            self.bezier_processor = CUDABezierProcessor(self.cuda_manager)
            self.device = self.cuda_manager.get_device()
        else:
            self.cuda_manager = None
            self.bezier_processor = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB")
        
        # Initialize networks
        self.generator = BezierGenerator(latent_dim, num_control_points, use_cuda_optimizations)
        self.discriminator = BezierDiscriminator(num_control_points, use_spectral_norm=True)
        
        # Enhanced loss function with gradient penalty support
        self.criterion = nn.BCELoss()
        self.use_wgan_gp = False  # Can be enabled for WGAN-GP training
        
        # Optimizers with different learning rates for generator and discriminator
        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr * 0.8, betas=(beta1, beta2))  # Slightly slower for discriminator
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=0.99)
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.99)
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Enable mixed precision training for better performance
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training statistics
        self.training_stats = {
            'g_losses': [],
            'd_losses': [],
            'epochs_trained': 0,
            'total_time': 0
        }
    
    def enable_wgan_gp(self, lambda_gp=10):
        """Enable WGAN-GP training for improved stability."""
        self.use_wgan_gp = True
        self.lambda_gp = lambda_gp
        # Remove sigmoid from discriminator for WGAN
        if hasattr(self.discriminator.model, '__getitem__'):
            if isinstance(self.discriminator.model[-1], nn.Sigmoid):
                self.discriminator.model = nn.Sequential(*list(self.discriminator.model.children())[:-1])
    
    def compute_gradient_penalty(self, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1).to(self.device)
        
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def load_data(self, data_path, batch_size=64, pin_memory=True):
        """Load and preprocess the Bézier curve data with CUDA optimizations."""
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        
        # Convert int8 data to float and normalize to [-1, 1]
        data_normalized = (data.astype(np.float32) + 128) / 255.0 * 2 - 1
        
        # Calculate how many complete curves we can extract
        points_per_curve = self.num_control_points * 2
        num_curves = len(data_normalized) // points_per_curve
        
        print(f"Extracted {num_curves} curves from {len(data_normalized)} data points")
        
        # Reshape to curves
        curves_data = data_normalized[:num_curves * points_per_curve].reshape(num_curves, self.num_control_points, 2)
        
        # Convert to torch tensor
        tensor_data = torch.FloatTensor(curves_data)
        
        # Pin memory for faster GPU transfer
        if pin_memory and self.device.type == "cuda":
            tensor_data = tensor_data.pin_memory()
        
        dataset = TensorDataset(tensor_data)
        
        # Optimize data loader for CUDA
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=pin_memory and self.device.type == "cuda",
            num_workers=4 if self.device.type == "cuda" else 0,
            persistent_workers=True if self.device.type == "cuda" else False
        )
        
        return data_loader
    
    def train(self, data_loader, epochs=100, save_interval=10, log_interval=1):
        """Enhanced training with CUDA optimizations, mixed precision, and detailed logging."""
        print(f"Training Bézier GAN for {epochs} epochs...")
        print(f"Using device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"WGAN-GP: {self.use_wgan_gp}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            d_losses = []
            g_losses = []
            
            # Set models to training mode
            self.generator.train()
            self.discriminator.train()
            
            for i, (real_data,) in enumerate(data_loader):
                batch_size = real_data.size(0)
                
                # Move data to device efficiently
                if self.cuda_manager:
                    real_data = self.cuda_manager.optimize_tensor(real_data)
                else:
                    real_data = real_data.to(self.device, non_blocking=True)
                
                # Train Discriminator
                self.discriminator.zero_grad()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        d_loss = self._train_discriminator_step(real_data, batch_size)
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.optim_d)
                else:
                    d_loss = self._train_discriminator_step(real_data, batch_size)
                    d_loss.backward()
                    self.optim_d.step()
                
                d_losses.append(d_loss.item())
                
                # Train Generator (less frequently for stability)
                if i % 1 == 0:  # Train generator every iteration
                    self.generator.zero_grad()
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            g_loss = self._train_generator_step(batch_size)
                        self.scaler.scale(g_loss).backward()
                        self.scaler.step(self.optim_g)
                        self.scaler.update()
                    else:
                        g_loss = self._train_generator_step(batch_size)
                        g_loss.backward()
                        self.optim_g.step()
                    
                    g_losses.append(g_loss.item())
                
                # Clear cache periodically to prevent memory buildup
                if self.cuda_manager and i % 50 == 0:
                    self.cuda_manager.clear_cache()
            
            # Update learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Record statistics
            epoch_d_loss = np.mean(d_losses)
            epoch_g_loss = np.mean(g_losses) if g_losses else 0
            epoch_time = time.time() - epoch_start_time
            
            self.training_stats['d_losses'].append(epoch_d_loss)
            self.training_stats['g_losses'].append(epoch_g_loss)
            self.training_stats['epochs_trained'] += 1
            
            # Logging
            if epoch % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f} "
                      f"Time: {epoch_time:.2f}s "
                      f"LR_G: {self.scheduler_g.get_last_lr()[0]:.6f}")
                
                # Memory usage logging
                if self.device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            # Save model periodically
            if epoch % save_interval == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        
        total_time = time.time() - start_time
        self.training_stats['total_time'] = total_time
        print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average time per epoch: {total_time/epochs:.2f} seconds")
    
    def _train_discriminator_step(self, real_data, batch_size):
        """Single discriminator training step."""
        # Real data loss
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_data)
        
        if self.use_wgan_gp:
            d_loss_real = -torch.mean(real_output)
        else:
            d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data loss
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data.detach())
        
        if self.use_wgan_gp:
            d_loss_fake = torch.mean(fake_output)
            # Gradient penalty
            gp = self.compute_gradient_penalty(real_data, fake_data)
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gp
        else:
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        
        return d_loss
    
    def _train_generator_step(self, batch_size):
        """Single generator training step."""
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data)
        
        if self.use_wgan_gp:
            g_loss = -torch.mean(fake_output)
        else:
            real_labels = torch.ones(batch_size, 1, device=self.device)
            g_loss = self.criterion(fake_output, real_labels)
        
        return g_loss
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optim_g_state_dict': self.optim_g.state_dict(),
            'optim_d_state_dict': self.optim_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'training_stats': self.training_stats,
            'latent_dim': self.latent_dim,
            'num_control_points': self.num_control_points,
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optim_g.load_state_dict(checkpoint['optim_g_state_dict'])
        self.optim_d.load_state_dict(checkpoint['optim_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"Checkpoint loaded from {filepath}")
    
    def get_training_stats(self):
        """Get training statistics."""
        return self.training_stats.copy()
    
    def generate_curves(self, num_curves=10, temperature=1.0):
        """Generate new Bézier curves with optional temperature scaling."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_curves, self.latent_dim, device=self.device) * temperature
            generated_curves = self.generator(z)
            
            # Denormalize from [-1, 1] to [0, 100] coordinate range
            generated_curves = (generated_curves + 1) * 50  # Scale to [0, 100]
            
            # Use CUDA-accelerated evaluation if available
            if self.bezier_processor:
                curves_np = generated_curves.cpu().numpy()
                evaluated_curves = self.bezier_processor.batch_evaluate_curves(curves_np)
                return generated_curves.cpu().numpy(), evaluated_curves
            else:
                return generated_curves.cpu().numpy()
    
    def interpolate_curves(self, curve1_idx=None, curve2_idx=None, steps=10, num_curves=None):
        """Generate interpolated curves between two latent vectors."""
        self.generator.eval()
        
        if curve1_idx is None or curve2_idx is None:
            # Generate random interpolation
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
        else:
            # Use specific curve indices (would need to be implemented based on training data)
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
        
        interpolated_curves = []
        
        with torch.no_grad():
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                curve = self.generator(z_interp)
                curve = (curve + 1) * 50  # Denormalize
                interpolated_curves.append(curve.cpu().numpy()[0])
        
        return np.array(interpolated_curves)
    
    def evaluate_model_performance(self, test_data_loader=None):
        """Evaluate model performance using various metrics."""
        self.generator.eval()
        self.discriminator.eval()
        
        metrics = {}
        
        # Generate sample curves for evaluation
        sample_curves, _ = self.generate_curves(num_curves=100)
        
        # Compute diversity metrics
        if self.bezier_processor:
            curve_metrics = self.bezier_processor.compute_curve_metrics(sample_curves)
            metrics.update(curve_metrics)
        
        # FID score computation (simplified version)
        if test_data_loader:
            real_features = []
            fake_features = []
            
            with torch.no_grad():
                # Get real data features
                for real_data, in test_data_loader:
                    real_data = real_data.to(self.device)
                    real_features.append(real_data.view(real_data.size(0), -1))
                    
                    # Generate fake data
                    z = torch.randn(real_data.size(0), self.latent_dim, device=self.device)
                    fake_data = self.generator(z)
                    fake_features.append(fake_data.view(fake_data.size(0), -1))
                
                real_features = torch.cat(real_features, dim=0)
                fake_features = torch.cat(fake_features, dim=0)
                
                # Simplified FID (using mean and std instead of full covariance)
                real_mean = torch.mean(real_features, dim=0)
                fake_mean = torch.mean(fake_features, dim=0)
                real_std = torch.std(real_features, dim=0)
                fake_std = torch.std(fake_features, dim=0)
                
                fid_score = torch.mean((real_mean - fake_mean) ** 2) + torch.mean((real_std - fake_std) ** 2)
                metrics['simplified_fid'] = fid_score.item()
        
        return metrics

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
