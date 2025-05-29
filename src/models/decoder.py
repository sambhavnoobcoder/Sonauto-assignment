"""
Hierarchical Decoder for Neural Audio Codec

This module implements a multi-scale decoder that reconstructs high-quality audio
from quantized hierarchical representations, with music-specific processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TransposedResidualBlock(nn.Module):
    """Residual block with transposed convolutions for upsampling"""
    
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


class MusicConditionedUpsampler(nn.Module):
    """Upsampler that incorporates music-specific conditioning"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        upsample_ratio: int,
        music_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.upsample_ratio = upsample_ratio
        self.music_dim = music_dim
        
        # Main upsampling path
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upsample_ratio * 2,
            stride=upsample_ratio,
            padding=upsample_ratio // 2
        )
        
        # Music conditioning
        if music_dim is not None:
            self.music_proj = nn.Conv1d(music_dim, out_channels, kernel_size=1)
            self.music_gate = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
        
        # Post-processing
        self.post_conv = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        music_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Upsample
        x = self.upsample_conv(x)
        
        # Apply music conditioning if available
        if music_features is not None and self.music_dim is not None:
            # Interpolate music features to match upsampled length
            music_upsampled = F.interpolate(
                music_features, size=x.shape[-1], mode='linear', align_corners=False
            )
            
            # Project and gate
            music_proj = self.music_proj(music_upsampled)
            combined = torch.cat([x, music_proj], dim=1)
            x = self.music_gate(combined) * x  # Gated residual connection
        
        # Post-processing
        x = self.post_conv(x)
        
        return x


class DecoderLevel(nn.Module):
    """Single level of the hierarchical decoder"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_ratio: int,
        music_dim: Optional[int] = None,
        use_music_features: bool = True
    ):
        super().__init__()
        
        self.upsample_ratio = upsample_ratio
        self.use_music_features = use_music_features
        
        # Music-conditioned upsampler
        self.upsampler = MusicConditionedUpsampler(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample_ratio=upsample_ratio,
            music_dim=music_dim if use_music_features else None
        )
        
        # Residual blocks with decreasing dilation
        self.residual_blocks = nn.ModuleList([
            TransposedResidualBlock(out_channels, dilation=2**(3-i))
            for i in range(4)
        ])
        
        # Final normalization
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        
    def forward(
        self, 
        x: torch.Tensor, 
        music_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Upsample with music conditioning
        x = self.upsampler(x, music_features)
        
        # Residual processing
        for block in self.residual_blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        x = self.activation(x)
        
        return x


class PerceptualEnhancer(nn.Module):
    """Enhance decoded audio using perceptual features"""
    
    def __init__(self, audio_channels: int, perceptual_dim: int = 128):
        super().__init__()
        
        # Perceptual feature processor
        self.perceptual_processor = nn.Sequential(
            nn.Conv1d(perceptual_dim, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, audio_channels, kernel_size=3, padding=1)
        )
        
        # Enhancement gate
        self.enhancement_gate = nn.Sequential(
            nn.Conv1d(audio_channels * 2, audio_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        audio: torch.Tensor, 
        perceptual_features: torch.Tensor
    ) -> torch.Tensor:
        # Process perceptual features
        perceptual_processed = self.perceptual_processor(perceptual_features)
        
        # Interpolate to match audio length
        perceptual_upsampled = F.interpolate(
            perceptual_processed, size=audio.shape[-1], 
            mode='linear', align_corners=False
        )
        
        # Gated enhancement
        combined = torch.cat([audio, perceptual_upsampled], dim=1)
        gate = self.enhancement_gate(combined)
        
        enhanced = audio + gate * perceptual_upsampled
        
        return enhanced


class HierarchicalDecoder(nn.Module):
    """
    Hierarchical decoder that reconstructs audio from quantized features
    
    Features:
    - Multi-scale upsampling with music conditioning
    - Perceptual enhancement
    - Progressive refinement across hierarchical levels
    """
    
    def __init__(
        self,
        decoder_dim: int = 512,
        channels: int = 2,
        compression_ratios: List[int] = [8, 16, 32],
        use_music_features: bool = True,
        perceptual_dim: int = 128,
        encoder_dim: int = None  # Add encoder_dim parameter
    ):
        super().__init__()
        
        self.decoder_dim = decoder_dim
        self.channels = channels
        self.compression_ratios = compression_ratios
        self.use_music_features = use_music_features
        
        # Store encoder_dim for dimension calculations
        if encoder_dim is None:
            encoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        
        # Reverse the compression ratios for decoding
        self.upsample_ratios = list(reversed(compression_ratios))
        
        # Input projection to map from codebook_dim to decoder_dim
        self.input_projection = nn.Conv1d(256, decoder_dim, kernel_size=1)  # codebook_dim=256 -> decoder_dim
        
        # Decoder levels
        self.decoder_levels = nn.ModuleList()
        current_dim = decoder_dim
        
        for i, ratio in enumerate(self.upsample_ratios):
            next_dim = decoder_dim // (2 ** i) if i < len(self.upsample_ratios) - 1 else decoder_dim // 4
            
            # Music feature dimension should match encoder output: encoder_level_dim // 4
            # Calculate corresponding encoder level dimension
            encoder_level_idx = len(self.upsample_ratios) - 1 - i  # Reverse mapping
            encoder_level_dim = encoder_dim // (2 ** (2 - encoder_level_idx))
            music_dim = encoder_level_dim // 4 if use_music_features else None
            
            level = DecoderLevel(
                in_channels=current_dim,
                out_channels=next_dim,
                upsample_ratio=ratio,
                music_dim=music_dim,
                use_music_features=use_music_features
            )
            
            self.decoder_levels.append(level)
            current_dim = next_dim
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(current_dim, current_dim // 2, kernel_size=7, padding=3),
            nn.GroupNorm(8, current_dim // 2),
            nn.GELU(),
            nn.Conv1d(current_dim // 2, channels, kernel_size=7, padding=3),
            nn.Tanh()  # Ensure output is in [-1, 1]
        )
        
        # Perceptual enhancer
        if use_music_features:
            self.perceptual_enhancer = PerceptualEnhancer(channels, perceptual_dim)
        
        # Skip connections for better gradient flow
        # Need to handle both forward pass (codebook_dim) and decompression (original encoder dims)
        encoder_output_dims = [
            encoder_dim // 4,  # Level 0: encoder_dim // 4
            encoder_dim // 2,  # Level 1: encoder_dim // 2
        ]
        
        # Calculate actual decoder level input dimensions
        decoder_level_dims = []
        current_dim = decoder_dim
        for i, ratio in enumerate(self.upsample_ratios):
            decoder_level_dims.append(current_dim)
            next_dim = decoder_dim // (2 ** i) if i < len(self.upsample_ratios) - 1 else decoder_dim // 4
            current_dim = next_dim
        
        # Create skip connections for both scenarios - output to match decoder level input dims
        self.skip_connections_forward = nn.ModuleList([
            nn.Conv1d(256, decoder_level_dims[i+1], kernel_size=1)  # For quantized features (codebook_dim=256)
            for i in range(len(compression_ratios) - 1)
        ])
        
        self.skip_connections_decompress = nn.ModuleList([
            nn.Conv1d(encoder_output_dims[i], decoder_level_dims[i+1], kernel_size=1)  # For decompression
            for i in range(min(len(encoder_output_dims), len(decoder_level_dims) - 1))
        ])
        
    def forward(
        self,
        quantized_features: List[torch.Tensor],
        music_features: Optional[List[torch.Tensor]] = None,
        perceptual_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode quantized features to audio
        
        Args:
            quantized_features: List of quantized features from hierarchical levels
            music_features: Optional list of music features for each level
            perceptual_features: Optional perceptual features for enhancement
            
        Returns:
            Reconstructed audio [batch, channels, time]
        """
        # Start from the deepest (most compressed) level
        x = quantized_features[-1]
        
        # Project from codebook_dim to decoder_dim
        x = self.input_projection(x)
        
        # Collect skip connections
        skip_features = []
        for i, feat in enumerate(quantized_features[:-1]):
            # Detect which skip connection to use based on feature dimensions
            feat_channels = feat.shape[1]
            
            # Quantized features always have codebook_dim channels (256)
            # We need to project them to the appropriate decoder level input dimensions
            if feat_channels == 256:  # codebook_dim
                # These are quantized features from the quantizer
                skip_proj = self.skip_connections_forward[i](feat)
            else:
                # These are original encoder features (for decompression scenario)
                skip_proj = self.skip_connections_decompress[i](feat)
            
            skip_features.append(skip_proj)
        
        # Progressive upsampling through decoder levels
        for i, decoder_level in enumerate(self.decoder_levels):
            # Get corresponding music features if available
            current_music_features = None
            if music_features is not None and i < len(music_features):
                # Use music features from corresponding encoder level
                music_idx = len(music_features) - 1 - i
                if music_idx >= 0:
                    current_music_features = music_features[music_idx]
            
            # Decode current level
            x = decoder_level(x, current_music_features)
            
            # Add skip connection if available
            if i < len(skip_features):
                skip_idx = len(skip_features) - 1 - i
                if skip_features[skip_idx].shape[-1] == x.shape[-1]:
                    x = x + skip_features[skip_idx]
        
        # Final output projection
        audio = self.output_proj(x)
        
        # Perceptual enhancement if features are available
        if perceptual_features is not None and hasattr(self, 'perceptual_enhancer'):
            audio = self.perceptual_enhancer(audio, perceptual_features)
        
        return audio
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the decoder"""
        receptive_field = 1
        
        for ratio in self.upsample_ratios:
            receptive_field = receptive_field * ratio + (ratio * 2 - 1)
        
        # Add contribution from residual blocks and final convolutions
        receptive_field += 7 + 5 + 7  # kernel sizes
        
        return receptive_field
    
    def estimate_output_length(self, input_lengths: List[int]) -> int:
        """
        Estimate output audio length given input feature lengths
        
        Args:
            input_lengths: List of feature lengths for each hierarchical level
            
        Returns:
            Estimated output audio length
        """
        # Start from the deepest level
        length = input_lengths[-1]
        
        # Apply upsampling ratios
        for ratio in self.upsample_ratios:
            length = length * ratio
        
        return length 