"""
Reconstruction Loss for Neural Audio Codec

This module implements the main reconstruction loss that combines multiple
loss components including perceptual, music-specific, and adversarial losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

from .perceptual import PerceptualLoss, SpectralLoss, MelSpectralLoss
from .adversarial import AdversarialLoss


class ReconstructionLoss(nn.Module):
    """
    Comprehensive reconstruction loss for neural audio codec
    
    Combines:
    - Time-domain reconstruction loss
    - Frequency-domain perceptual losses
    - Music-specific losses
    - Adversarial losses
    - Vector quantization losses
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        # Loss weights
        time_weight: float = 1.0,
        spectral_weight: float = 1.0,
        mel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        music_weight: float = 0.5,
        adversarial_weight: float = 0.1,
        commitment_weight: float = 0.25,
        # Loss configurations
        use_adversarial: bool = True,
        use_music_loss: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Loss weights
        self.time_weight = time_weight
        self.spectral_weight = spectral_weight
        self.mel_weight = mel_weight
        self.perceptual_weight = perceptual_weight
        self.music_weight = music_weight
        self.adversarial_weight = adversarial_weight
        self.commitment_weight = commitment_weight
        
        self.use_adversarial = use_adversarial
        self.use_music_loss = use_music_loss
        
        # Initialize loss components
        self.spectral_loss = SpectralLoss(
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate
        )
        
        self.mel_loss = MelSpectralLoss(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=128
        )
        
        self.perceptual_loss = PerceptualLoss(
            sample_rate=sample_rate
        )
        
        if use_music_loss:
            from .music import MusicLoss
            self.music_loss = MusicLoss(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )
        
        if use_adversarial:
            self.adversarial_loss = AdversarialLoss()
    
    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        discriminator_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive reconstruction loss
        
        Args:
            original: Original audio [batch, channels, time]
            reconstructed: Reconstructed audio [batch, channels, time]
            model_outputs: Dictionary containing model outputs (codes, losses, etc.)
            discriminator_outputs: Optional discriminator outputs for adversarial loss
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Time-domain reconstruction loss
        time_loss = F.l1_loss(reconstructed, original)
        losses['time_loss'] = time_loss
        
        # Spectral reconstruction loss
        spectral_loss = self.spectral_loss(reconstructed, original)
        losses['spectral_loss'] = spectral_loss
        
        # Mel-spectral loss
        mel_loss = self.mel_loss(reconstructed, original)
        losses['mel_loss'] = mel_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(reconstructed, original)
        losses['perceptual_loss'] = perceptual_loss
        
        # Music-specific loss
        if self.use_music_loss and hasattr(self, 'music_loss'):
            music_loss = self.music_loss(reconstructed, original)
            losses['music_loss'] = music_loss
        else:
            losses['music_loss'] = torch.tensor(0.0, device=original.device)
        
        # Vector quantization losses
        commitment_loss = model_outputs.get('commitment_loss', torch.tensor(0.0, device=original.device))
        codebook_loss = model_outputs.get('codebook_loss', torch.tensor(0.0, device=original.device))
        
        losses['commitment_loss'] = commitment_loss
        losses['codebook_loss'] = codebook_loss
        
        # Adversarial loss
        if self.use_adversarial and discriminator_outputs is not None:
            adv_loss = self.adversarial_loss(discriminator_outputs)
            losses['adversarial_loss'] = adv_loss
        else:
            losses['adversarial_loss'] = torch.tensor(0.0, device=original.device)
        
        # Compute total loss
        total_loss = (
            self.time_weight * time_loss +
            self.spectral_weight * spectral_loss +
            self.mel_weight * mel_loss +
            self.perceptual_weight * perceptual_loss +
            self.music_weight * losses['music_loss'] +
            self.commitment_weight * commitment_loss +
            self.adversarial_weight * losses['adversarial_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics
        
        Args:
            original: Original audio [batch, channels, time]
            reconstructed: Reconstructed audio [batch, channels, time]
            
        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = {}
        
        # Signal-to-Noise Ratio (SNR)
        signal_power = torch.mean(original ** 2, dim=-1, keepdim=True)
        noise_power = torch.mean((original - reconstructed) ** 2, dim=-1, keepdim=True)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        metrics['snr_db'] = torch.mean(snr)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = torch.mean((original - reconstructed) ** 2)
        psnr = 20 * torch.log10(torch.max(torch.abs(original)) / torch.sqrt(mse + 1e-8))
        metrics['psnr_db'] = psnr
        
        # Spectral convergence
        original_stft = torch.stft(
            original.view(-1, original.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        reconstructed_stft = torch.stft(
            reconstructed.view(-1, reconstructed.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        
        original_mag = torch.abs(original_stft)
        reconstructed_mag = torch.abs(reconstructed_stft)
        
        spectral_convergence = torch.norm(original_mag - reconstructed_mag, p='fro') / torch.norm(original_mag, p='fro')
        metrics['spectral_convergence'] = spectral_convergence
        
        # Log spectral distance
        log_spectral_distance = torch.mean(
            (torch.log(original_mag + 1e-8) - torch.log(reconstructed_mag + 1e-8)) ** 2
        )
        metrics['log_spectral_distance'] = log_spectral_distance
        
        return metrics
    
    def update_weights(self, epoch: int, total_epochs: int):
        """
        Update loss weights during training (curriculum learning)
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
        """
        # Gradually increase adversarial loss weight
        if self.use_adversarial:
            progress = epoch / total_epochs
            self.adversarial_weight = min(0.1, 0.01 + 0.09 * progress)
        
        # Gradually increase music loss weight
        if self.use_music_loss:
            progress = epoch / total_epochs
            self.music_weight = min(0.5, 0.1 + 0.4 * progress)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return {
            'time_weight': self.time_weight,
            'spectral_weight': self.spectral_weight,
            'mel_weight': self.mel_weight,
            'perceptual_weight': self.perceptual_weight,
            'music_weight': self.music_weight,
            'adversarial_weight': self.adversarial_weight,
            'commitment_weight': self.commitment_weight
        } 