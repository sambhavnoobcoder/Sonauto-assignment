"""
Perceptual Loss Functions for Neural Audio Codec

This module implements various perceptual loss functions that capture
human auditory perception characteristics for better audio reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np


class SpectralLoss(nn.Module):
    """
    Multi-resolution spectral loss that compares STFT magnitudes
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        sample_rate: int = 44100,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_ratios: List[float] = [0.25, 0.25, 0.25],
        win_length_ratio: float = 1.0
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.fft_sizes = fft_sizes
        self.hop_ratios = hop_ratios
        self.win_length_ratio = win_length_ratio
        
        # Create windows for each FFT size
        self.windows = nn.ParameterDict()
        for fft_size in fft_sizes:
            win_length = int(fft_size * win_length_ratio)
            window = torch.hann_window(win_length)
            self.windows[str(fft_size)] = nn.Parameter(window, requires_grad=False)
    
    def stft_magnitude(self, audio: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        """Compute STFT magnitude"""
        window = self.windows[str(n_fft)]
        
        stft = torch.stft(
            audio.view(-1, audio.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        return magnitude
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution spectral loss
        
        Args:
            pred: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
            
        Returns:
            Spectral loss value
        """
        total_loss = 0.0
        
        for fft_size, hop_ratio in zip(self.fft_sizes, self.hop_ratios):
            hop_length = int(fft_size * hop_ratio)
            
            # Compute STFT magnitudes
            pred_mag = self.stft_magnitude(pred, fft_size, hop_length)
            target_mag = self.stft_magnitude(target, fft_size, hop_length)
            
            # Spectral convergence loss
            spectral_conv_loss = torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')
            
            # Log magnitude loss
            log_mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-7),
                torch.log(target_mag + 1e-7)
            )
            
            total_loss += spectral_conv_loss + log_mag_loss
        
        return total_loss / len(self.fft_sizes)


class MelSpectralLoss(nn.Module):
    """
    Mel-scale spectral loss for perceptually relevant frequency analysis
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        # Create mel filter bank
        mel_filters = self._create_mel_filters()
        self.register_buffer('mel_filters', mel_filters)
        
        # Window for STFT
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
    
    def _create_mel_filters(self) -> torch.Tensor:
        """Create mel filter bank"""
        # Convert to mel scale
        mel_min = 2595 * np.log10(1 + self.f_min / 700)
        mel_max = 2595 * np.log10(1 + self.f_max / 700)
        
        # Create mel points
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        
        # Convert to FFT bin indices
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filter bank
        filters = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        
        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            # Left slope
            for j in range(left, center):
                if center != left:
                    filters[i - 1, j] = (j - left) / (center - left)
            
            # Right slope
            for j in range(center, right):
                if right != center:
                    filters[i - 1, j] = (right - j) / (right - center)
        
        return filters
    
    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram"""
        # Compute STFT
        stft = torch.stft(
            audio.view(-1, audio.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        # Get magnitude
        magnitude = torch.abs(stft)  # [batch*channels, freq, time]
        
        # Apply mel filters
        mel_spec = torch.matmul(self.mel_filters, magnitude)
        
        return mel_spec
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute mel-spectral loss
        
        Args:
            pred: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
            
        Returns:
            Mel-spectral loss value
        """
        # Compute mel spectrograms
        pred_mel = self.compute_mel_spectrogram(pred)
        target_mel = self.compute_mel_spectrogram(target)
        
        # L1 loss on mel spectrograms
        mel_loss = F.l1_loss(pred_mel, target_mel)
        
        # Log mel loss for better perceptual alignment
        log_mel_loss = F.l1_loss(
            torch.log(pred_mel + 1e-7),
            torch.log(target_mel + 1e-7)
        )
        
        return mel_loss + log_mel_loss


class PerceptualLoss(nn.Module):
    """
    Comprehensive perceptual loss combining multiple perceptual metrics
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        use_spectral: bool = True,
        use_mel: bool = True,
        use_mfcc: bool = True,
        spectral_weight: float = 1.0,
        mel_weight: float = 1.0,
        mfcc_weight: float = 0.5
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.use_spectral = use_spectral
        self.use_mel = use_mel
        self.use_mfcc = use_mfcc
        
        self.spectral_weight = spectral_weight
        self.mel_weight = mel_weight
        self.mfcc_weight = mfcc_weight
        
        # Initialize loss components
        if use_spectral:
            self.spectral_loss = SpectralLoss(sample_rate=sample_rate)
        
        if use_mel:
            self.mel_loss = MelSpectralLoss(sample_rate=sample_rate)
        
        if use_mfcc:
            self.mfcc_loss = MFCCLoss(sample_rate=sample_rate)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute comprehensive perceptual loss
        
        Args:
            pred: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
            
        Returns:
            Combined perceptual loss
        """
        total_loss = 0.0
        
        if self.use_spectral:
            spectral_loss = self.spectral_loss(pred, target)
            total_loss += self.spectral_weight * spectral_loss
        
        if self.use_mel:
            mel_loss = self.mel_loss(pred, target)
            total_loss += self.mel_weight * mel_loss
        
        if self.use_mfcc:
            mfcc_loss = self.mfcc_loss(pred, target)
            total_loss += self.mfcc_weight * mfcc_loss
        
        return total_loss


class MFCCLoss(nn.Module):
    """
    MFCC-based perceptual loss for timbral similarity
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Create mel spectrogram layer
        self.mel_spec = MelSpectralLoss(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # DCT matrix for MFCC computation
        dct_matrix = self._create_dct_matrix()
        self.register_buffer('dct_matrix', dct_matrix)
    
    def _create_dct_matrix(self) -> torch.Tensor:
        """Create DCT matrix for MFCC computation"""
        dct_matrix = torch.zeros(self.n_mfcc, self.n_mels)
        
        for i in range(self.n_mfcc):
            for j in range(self.n_mels):
                dct_matrix[i, j] = math.cos(math.pi * i * (j + 0.5) / self.n_mels)
        
        # Normalize
        dct_matrix[0] *= math.sqrt(1.0 / self.n_mels)
        dct_matrix[1:] *= math.sqrt(2.0 / self.n_mels)
        
        return dct_matrix
    
    def compute_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute MFCC features"""
        # Get mel spectrogram
        mel_spec = self.mel_spec.compute_mel_spectrogram(audio)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-7)
        
        # Apply DCT
        mfcc = torch.matmul(self.dct_matrix, log_mel)
        
        return mfcc
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC loss
        
        Args:
            pred: Predicted audio [batch, channels, time]
            target: Target audio [batch, channels, time]
            
        Returns:
            MFCC loss value
        """
        # Compute MFCC features
        pred_mfcc = self.compute_mfcc(pred)
        target_mfcc = self.compute_mfcc(target)
        
        # L2 loss on MFCC features
        mfcc_loss = F.mse_loss(pred_mfcc, target_mfcc)
        
        return mfcc_loss 