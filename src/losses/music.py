"""
Music-specific Loss Functions for Neural Audio Codec

This module implements loss functions specifically designed for music audio,
focusing on harmonic, rhythmic, and timbral characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class HarmonicLoss(nn.Module):
    """Loss function that emphasizes harmonic content preservation"""
    
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute harmonic loss"""
        # Simple spectral loss for now
        pred_stft = torch.stft(pred.view(-1, pred.shape[-1]), self.n_fft, self.hop_length, return_complex=True)
        target_stft = torch.stft(target.view(-1, target.shape[-1]), self.n_fft, self.hop_length, return_complex=True)
        
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        return F.l1_loss(pred_mag, target_mag)


class RhythmicLoss(nn.Module):
    """Loss function that emphasizes rhythmic pattern preservation"""
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute rhythmic loss"""
        # Simple time-domain loss for now
        return F.l1_loss(pred, target)


class TimbrePreservationLoss(nn.Module):
    """Loss function that preserves timbral characteristics"""
    
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute timbre preservation loss"""
        # Simple spectral loss for now
        pred_stft = torch.stft(pred.view(-1, pred.shape[-1]), self.n_fft, self.hop_length, return_complex=True)
        target_stft = torch.stft(target.view(-1, target.shape[-1]), self.n_fft, self.hop_length, return_complex=True)
        
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        return F.mse_loss(pred_mag, target_mag)


class MusicLoss(nn.Module):
    """Comprehensive music loss combining harmonic, rhythmic, and timbral losses"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        harmonic_weight: float = 1.0,
        rhythmic_weight: float = 1.0,
        timbre_weight: float = 1.0
    ):
        super().__init__()
        
        self.harmonic_loss = HarmonicLoss(sample_rate, n_fft, hop_length)
        self.rhythmic_loss = RhythmicLoss(sample_rate)
        self.timbre_loss = TimbrePreservationLoss(sample_rate, n_fft, hop_length)
        
        self.harmonic_weight = harmonic_weight
        self.rhythmic_weight = rhythmic_weight
        self.timbre_weight = timbre_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute total music loss"""
        harmonic_loss = self.harmonic_loss(pred, target)
        rhythmic_loss = self.rhythmic_loss(pred, target)
        timbre_loss = self.timbre_loss(pred, target)
        
        total_loss = (
            self.harmonic_weight * harmonic_loss +
            self.rhythmic_weight * rhythmic_loss +
            self.timbre_weight * timbre_loss
        )
        
        return total_loss 