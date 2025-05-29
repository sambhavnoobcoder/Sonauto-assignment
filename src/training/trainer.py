"""
Training Script for Neural Audio Codec

This module provides a comprehensive trainer for the neural audio codec,
integrating reconstruction, perceptual, music-specific, and adversarial losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

# Fix imports to work from outside package
try:
    from ..models.codec import NeuralAudioCodec
    from ..losses import (
        ReconstructionLoss, PerceptualLoss, MusicLoss, AdversarialLoss,
        MultiScaleDiscriminator, HybridDiscriminator
    )
except ImportError:
    # Fallback for when imported from outside package
    from models.codec import NeuralAudioCodec
    from losses import (
        ReconstructionLoss, PerceptualLoss, MusicLoss, AdversarialLoss,
        MultiScaleDiscriminator, HybridDiscriminator
    )


class AudioCodecTrainer:
    """
    Comprehensive trainer for neural audio codec
    
    Handles training with multiple loss functions, adversarial training,
    and comprehensive logging and checkpointing.
    """
    
    def __init__(
        self,
        model: NeuralAudioCodec,
        config: Dict[str, Any],
        device: torch.device = None,
        experiment_dir: str = "experiments"
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_dir = Path(experiment_dir)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize discriminator for adversarial training
        if config.get('use_adversarial', True):
            self.discriminator = HybridDiscriminator(
                channels=config.get('channels', 2),
                time_scales=config.get('discriminator_scales', [1, 2, 4]),
                use_spectral=config.get('use_spectral_discriminator', True)
            ).to(self.device)
        else:
            self.discriminator = None
        
        # Initialize loss functions
        self._init_losses()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Setup logging
        self._setup_logging()
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def _init_losses(self):
        """Initialize all loss functions"""
        loss_config = self.config.get('losses', {})
        
        # Individual loss components instead of comprehensive reconstruction loss
        # Time domain loss (L1)
        self.l1_weight = loss_config.get('l1_weight', 1.0)
        self.l2_weight = loss_config.get('l2_weight', 0.0)
        
        # Spectral losses
        try:
            from ..losses.perceptual import SpectralLoss, MelSpectralLoss
        except ImportError:
            from losses.perceptual import SpectralLoss, MelSpectralLoss
        
        self.spectral_loss = SpectralLoss()
        self.mel_loss = MelSpectralLoss()
        self.stft_weight = loss_config.get('stft_weight', 1.0)
        self.mel_weight = loss_config.get('mel_weight', 1.0)
        
        # Perceptual loss
        if loss_config.get('use_perceptual', True):
            self.perceptual_loss = PerceptualLoss(
                use_spectral=loss_config.get('use_spectral_loss', True),
                use_mel=loss_config.get('use_mel_loss', True),
                use_mfcc=loss_config.get('use_mfcc_loss', True),
                spectral_weight=loss_config.get('spectral_weight', 1.0),
                mel_weight=loss_config.get('mel_weight', 1.0),
                mfcc_weight=loss_config.get('mfcc_weight', 0.5)
            )
        else:
            self.perceptual_loss = None
        
        # Music-specific loss
        if loss_config.get('use_music', True):
            self.music_loss = MusicLoss(
                harmonic_weight=loss_config.get('harmonic_weight', 1.0),
                rhythmic_weight=loss_config.get('rhythmic_weight', 1.0),
                timbre_weight=loss_config.get('timbre_weight', 1.0)
            )
        else:
            self.music_loss = None
        
        # Adversarial loss
        if self.discriminator is not None:
            self.adversarial_loss = AdversarialLoss(
                loss_type=loss_config.get('adversarial_type', 'hinge'),
                feature_matching_weight=loss_config.get('feature_matching_weight', 10.0),
                use_feature_matching=loss_config.get('use_feature_matching', True)
            )
        else:
            self.adversarial_loss = None
        
        # Loss weights
        self.loss_weights = {
            'reconstruction': loss_config.get('reconstruction_weight', 1.0),
            'perceptual': loss_config.get('perceptual_weight', 1.0),
            'music': loss_config.get('music_weight', 1.0),
            'adversarial': loss_config.get('adversarial_weight', 0.1),
            'feature_matching': loss_config.get('feature_matching_weight', 10.0)
        }
    
    def _init_optimizers(self):
        """Initialize optimizers for generator and discriminator"""
        opt_config = self.config.get('optimizer', {})
        
        # Generator optimizer
        self.gen_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config.get('gen_lr', 1e-4),
            betas=opt_config.get('betas', (0.8, 0.99)),
            weight_decay=opt_config.get('weight_decay', 1e-4)
        )
        
        # Discriminator optimizer
        if self.discriminator is not None:
            self.disc_optimizer = optim.AdamW(
                self.discriminator.parameters(),
                lr=opt_config.get('disc_lr', 1e-4),
                betas=opt_config.get('betas', (0.8, 0.99)),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        
        # Learning rate schedulers
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('use_scheduler', True):
            self.gen_scheduler = optim.lr_scheduler.ExponentialLR(
                self.gen_optimizer,
                gamma=scheduler_config.get('gamma', 0.999)
            )
            
            if self.discriminator is not None:
                self.disc_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.disc_optimizer,
                    gamma=scheduler_config.get('gamma', 0.999)
                )
        else:
            self.gen_scheduler = None
            self.disc_scheduler = None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.experiment_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def compute_generator_loss(
        self,
        audio: torch.Tensor,
        reconstructed: torch.Tensor,
        quantized_features: List[torch.Tensor],
        commitment_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total generator loss
        
        Args:
            audio: Original audio [batch, channels, time]
            reconstructed: Reconstructed audio [batch, channels, time]
            quantized_features: List of quantized features
            commitment_loss: VQ commitment loss
            
        Returns:
            Total loss and loss components dictionary
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Time-domain reconstruction losses
        l1_loss = F.l1_loss(reconstructed, audio)
        losses['l1'] = l1_loss.item()
        total_loss += self.l1_weight * l1_loss
        
        if self.l2_weight > 0:
            l2_loss = F.mse_loss(reconstructed, audio)
            losses['l2'] = l2_loss.item()
            total_loss += self.l2_weight * l2_loss
        
        # Spectral losses
        spectral_loss = self.spectral_loss(reconstructed, audio)
        losses['spectral'] = spectral_loss.item()
        total_loss += self.stft_weight * spectral_loss
        
        mel_loss = self.mel_loss(reconstructed, audio)
        losses['mel'] = mel_loss.item()
        total_loss += self.mel_weight * mel_loss
        
        # Commitment loss from VQ
        losses['commitment'] = commitment_loss.item()
        total_loss += commitment_loss
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perceptual_loss = self.perceptual_loss(reconstructed, audio)
            losses['perceptual'] = perceptual_loss.item()
            total_loss += self.loss_weights['perceptual'] * perceptual_loss
        
        # Music-specific loss
        if self.music_loss is not None:
            music_loss = self.music_loss(reconstructed, audio)
            losses['music'] = music_loss.item()
            total_loss += self.loss_weights['music'] * music_loss
        
        # Adversarial loss
        if self.discriminator is not None and self.adversarial_loss is not None:
            # Get discriminator outputs for fake samples
            fake_outputs = self.discriminator(reconstructed)
            
            # Flatten outputs for adversarial loss computation
            if isinstance(fake_outputs, dict):
                all_fake_outputs = []
                for key, outputs in fake_outputs.items():
                    all_fake_outputs.extend(outputs)
            else:
                all_fake_outputs = fake_outputs
            
            # Compute adversarial loss
            adv_loss = self.adversarial_loss.generator_loss(all_fake_outputs)
            losses['adversarial'] = adv_loss.item()
            total_loss += self.loss_weights['adversarial'] * adv_loss
            
            # Feature matching loss
            if self.adversarial_loss.use_feature_matching:
                real_outputs = self.discriminator(audio.detach())
                if isinstance(real_outputs, dict):
                    all_real_outputs = []
                    for key, outputs in real_outputs.items():
                        all_real_outputs.extend(outputs)
                else:
                    all_real_outputs = real_outputs
                
                fm_loss = self.adversarial_loss.feature_matching_loss(
                    all_real_outputs, all_fake_outputs
                )
                losses['feature_matching'] = fm_loss.item()
                total_loss += self.loss_weights['feature_matching'] * fm_loss
        
        losses['total'] = total_loss.item()
        return total_loss, losses
    
    def compute_discriminator_loss(
        self,
        real_audio: torch.Tensor,
        fake_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss
        
        Args:
            real_audio: Real audio samples
            fake_audio: Generated audio samples
            
        Returns:
            Discriminator loss and loss components
        """
        if self.discriminator is None or self.adversarial_loss is None:
            return torch.tensor(0.0), {}
        
        # Get discriminator outputs
        real_outputs = self.discriminator(real_audio)
        fake_outputs = self.discriminator(fake_audio.detach())
        
        # Flatten outputs
        if isinstance(real_outputs, dict):
            all_real_outputs = []
            all_fake_outputs = []
            for key in real_outputs.keys():
                all_real_outputs.extend(real_outputs[key])
                all_fake_outputs.extend(fake_outputs[key])
        else:
            all_real_outputs = real_outputs
            all_fake_outputs = fake_outputs
        
        # Compute discriminator loss
        disc_loss = self.adversarial_loss.discriminator_loss(
            all_real_outputs, all_fake_outputs
        )
        
        return disc_loss, {'discriminator': disc_loss.item()}
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Audio batch [batch, channels, time]
            
        Returns:
            Dictionary of loss values
        """
        batch = batch.to(self.device)
        
        # Forward pass through codec
        result = self.model(batch)
        reconstructed = result['audio']
        quantized_features = result['quantized']
        commitment_loss = result['commitment_loss']
        
        # Train discriminator
        disc_losses = {}
        if self.discriminator is not None:
            self.disc_optimizer.zero_grad()
            disc_loss, disc_losses = self.compute_discriminator_loss(batch, reconstructed)
            if disc_loss.item() > 0:
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.disc_optimizer.step()
        
        # Train generator
        self.gen_optimizer.zero_grad()
        gen_loss, gen_losses = self.compute_generator_loss(
            batch, reconstructed, quantized_features, commitment_loss
        )
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.gen_optimizer.step()
        
        # Combine losses
        all_losses = {**gen_losses, **disc_losses}
        
        return all_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation loop
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                # Forward pass
                result = self.model(batch)
                reconstructed = result['audio']
                quantized_features = result['quantized']
                commitment_loss = result['commitment_loss']
                
                # Compute losses
                _, gen_losses = self.compute_generator_loss(
                    batch, reconstructed, quantized_features, commitment_loss
                )
                
                if self.discriminator is not None:
                    _, disc_losses = self.compute_discriminator_loss(batch, reconstructed)
                    gen_losses.update(disc_losses)
                
                # Accumulate losses
                for key, value in gen_losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value
                
                num_batches += 1
        
        # Average losses
        avg_losses = {f"val_{key}": value / num_batches for key, value in total_losses.items()}
        
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['disc_optimizer_state_dict'] = self.disc_optimizer.state_dict()
        
        if self.gen_scheduler is not None:
            checkpoint['gen_scheduler_state_dict'] = self.gen_scheduler.state_dict()
        
        if self.disc_scheduler is not None:
            checkpoint['disc_scheduler_state_dict'] = self.disc_scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.experiment_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.discriminator is not None and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        if self.gen_scheduler is not None and 'gen_scheduler_state_dict' in checkpoint:
            self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler_state_dict'])
        
        if self.disc_scheduler is not None and 'disc_scheduler_state_dict' in checkpoint:
            self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_every: int = 10,
        validate_every: int = 5
    ):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.discriminator is not None:
            self.logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            if self.discriminator is not None:
                self.discriminator.train()
            
            epoch_losses = {}
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                step_losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in step_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value
                
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                if num_batches % 10 == 0:
                    current_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                    progress_bar.set_postfix(current_losses)
            
            # Average epoch losses
            avg_epoch_losses = {key: value / num_batches for key, value in epoch_losses.items()}
            
            # Validation
            val_losses = {}
            if val_loader is not None and (epoch + 1) % validate_every == 0:
                val_losses = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.gen_scheduler is not None:
                self.gen_scheduler.step()
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            
            for key, value in avg_epoch_losses.items():
                self.logger.info(f"Train {key}: {value:.6f}")
            
            for key, value in val_losses.items():
                self.logger.info(f"{key}: {value:.6f}")
            
            # Save checkpoint
            current_val_loss = val_losses.get('val_total', avg_epoch_losses.get('total', float('inf')))
            is_best = current_val_loss < self.best_loss
            if is_best:
                self.best_loss = current_val_loss
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            self.current_epoch = epoch + 1
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")


def create_trainer_from_config(config_path: str, model: NeuralAudioCodec) -> AudioCodecTrainer:
    """
    Create trainer from configuration file
    
    Args:
        config_path: Path to configuration JSON file
        model: Neural audio codec model
        
    Returns:
        Configured trainer instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return AudioCodecTrainer(model, config) 