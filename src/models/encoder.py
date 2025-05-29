"""
Hierarchical Encoder for Neural Audio Codec

This module implements a multi-scale encoder that captures audio features
at different temporal resolutions, with special attention to musical characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions for temporal modeling"""
    
    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.activation(x + residual)


class MusicFeatureExtractor(nn.Module):
    """Extract music-specific features like harmony, rhythm, and timbre"""
    
    def __init__(self, input_dim: int, feature_dim: int = 128):
        super().__init__()
        
        # Harmonic feature extraction (focus on pitch relationships)
        self.harmonic_conv = nn.Sequential(
            nn.Conv1d(input_dim, feature_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, feature_dim),
            nn.GELU()
        )
        
        # Rhythmic feature extraction (focus on temporal patterns)
        self.rhythmic_conv = nn.Sequential(
            nn.Conv1d(input_dim, feature_dim, kernel_size=15, padding=7, dilation=2),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=11, padding=5, dilation=3),
            nn.GroupNorm(8, feature_dim),
            nn.GELU()
        )
        
        # Timbral feature extraction (focus on spectral characteristics)
        self.timbral_conv = nn.Sequential(
            nn.Conv1d(input_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU()
        )
        
        # Feature fusion
        self.fusion = nn.Conv1d(feature_dim * 3, feature_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        harmonic = self.harmonic_conv(x)
        rhythmic = self.rhythmic_conv(x)
        timbral = self.timbral_conv(x)
        
        # Ensure all features have the same temporal dimension
        min_length = min(harmonic.shape[-1], rhythmic.shape[-1], timbral.shape[-1])
        
        harmonic = harmonic[..., :min_length]
        rhythmic = rhythmic[..., :min_length]
        timbral = timbral[..., :min_length]
        
        # Concatenate and fuse features
        combined = torch.cat([harmonic, rhythmic, timbral], dim=1)
        return self.fusion(combined)


class EncoderLevel(nn.Module):
    """Single level of the hierarchical encoder"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        compression_ratio: int,
        use_music_features: bool = True
    ):
        super().__init__()
        
        self.compression_ratio = compression_ratio
        self.use_music_features = use_music_features
        
        # Initial convolution
        self.input_conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=7, padding=3
        )
        
        # Residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(out_channels, dilation=2**i)
            for i in range(4)
        ])
        
        # Downsampling convolution
        self.downsample = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=compression_ratio * 2,
            stride=compression_ratio,
            padding=compression_ratio // 2
        )
        
        # Music feature extractor
        if use_music_features:
            self.music_extractor = MusicFeatureExtractor(out_channels, out_channels // 4)
        
        # Normalization and activation
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Initial processing
        x = self.input_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # Residual processing
        for block in self.residual_blocks:
            x = block(x)
        
        # Extract music features before downsampling
        music_features = None
        if self.use_music_features:
            music_features = self.music_extractor(x)
        
        # Downsample
        encoded = self.downsample(x)
        
        return {
            'encoded': encoded,
            'music_features': music_features
        }


class PerceptualFeatureExtractor(nn.Module):
    """Extract perceptual features for loss computation"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Multi-scale feature extraction
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=k, padding=k//2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU()
            ) for k in [3, 7, 15, 31]
        ])
        
        # Feature aggregation
        self.aggregator = nn.Conv1d(32 * 4, 128, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_features = []
        for scale in self.scales:
            scale_features.append(scale(x))
        
        combined = torch.cat(scale_features, dim=1)
        return self.aggregator(combined)


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder that processes audio at multiple temporal scales
    
    Features:
    - Multi-scale temporal modeling
    - Music-specific feature extraction
    - Perceptual feature computation for loss functions
    """
    
    def __init__(
        self,
        channels: int = 2,
        encoder_dim: int = 512,
        compression_ratios: List[int] = [8, 16, 32],
        use_music_features: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.encoder_dim = encoder_dim
        self.compression_ratios = compression_ratios
        self.use_music_features = use_music_features
        
        # Input projection
        self.input_proj = nn.Conv1d(channels, encoder_dim // 4, kernel_size=7, padding=3)
        
        # Hierarchical encoder levels
        self.encoder_levels = nn.ModuleList()
        current_dim = encoder_dim // 4
        
        for i, ratio in enumerate(compression_ratios):
            next_dim = encoder_dim // (2 ** (2 - i))  # Increase capacity at deeper levels
            
            level = EncoderLevel(
                in_channels=current_dim,
                out_channels=next_dim,
                compression_ratio=ratio,
                use_music_features=use_music_features
            )
            
            self.encoder_levels.append(level)
            current_dim = next_dim
        
        # Perceptual feature extractor
        self.perceptual_extractor = PerceptualFeatureExtractor(encoder_dim // 4)
        
        # Final normalization
        self.final_norm = nn.GroupNorm(8, current_dim)
        
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode audio hierarchically
        
        Args:
            audio: Input audio [batch, channels, time]
            
        Returns:
            Dictionary containing hierarchical features and music features
        """
        # Input projection
        x = self.input_proj(audio)
        
        # Extract perceptual features from input representation
        perceptual_features = self.perceptual_extractor(x)
        
        # Hierarchical encoding
        hierarchical_features = []
        music_features = []
        
        for level in self.encoder_levels:
            level_output = level(x)
            
            hierarchical_features.append(level_output['encoded'])
            
            if level_output['music_features'] is not None:
                music_features.append(level_output['music_features'])
            
            x = level_output['encoded']
        
        # Final normalization
        x = self.final_norm(x)
        hierarchical_features[-1] = x
        
        result = {
            'hierarchical_features': hierarchical_features,
            'perceptual_features': perceptual_features
        }
        
        if music_features:
            result['music_features'] = music_features
        
        return result
    
    def get_output_shapes(self, input_length: int) -> List[Tuple[int, int]]:
        """
        Calculate output shapes for each hierarchical level
        
        Args:
            input_length: Length of input audio
            
        Returns:
            List of (channels, length) tuples for each level
        """
        shapes = []
        current_length = input_length
        current_dim = self.encoder_dim // 4
        
        for i, ratio in enumerate(self.compression_ratios):
            current_length = current_length // ratio
            current_dim = self.encoder_dim // (2 ** (2 - i))
            shapes.append((current_dim, current_length))
        
        return shapes 