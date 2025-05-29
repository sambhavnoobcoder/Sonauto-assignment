"""
Adversarial Loss Functions for Neural Audio Codec

This module implements adversarial training components including
multi-scale discriminators and adversarial losses for improved perceptual quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for adversarial training
    
    Uses multiple discriminators operating at different temporal scales
    to capture both local and global audio characteristics.
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        channels: int = 2,
        base_channels: int = 64,
        max_channels: int = 512,
        kernel_sizes: List[int] = [15, 41, 5, 3],
        groups: List[int] = [1, 4, 16, 64]
    ):
        super().__init__()
        
        self.scales = scales
        self.discriminators = nn.ModuleList()
        
        # Create discriminator for each scale
        for scale in scales:
            discriminator = ScaleDiscriminator(
                channels=channels,
                base_channels=base_channels,
                max_channels=max_channels,
                kernel_sizes=kernel_sizes,
                groups=groups
            )
            self.discriminators.append(discriminator)
        
        # Pooling layers for different scales
        self.pooling = nn.ModuleList()
        for scale in scales[1:]:  # Skip scale 1 (no pooling)
            self.pooling.append(nn.AvgPool1d(kernel_size=scale, stride=scale))
    
    def forward(self, audio: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass through multi-scale discriminator
        
        Args:
            audio: Input audio [batch, channels, time]
            
        Returns:
            List of discriminator outputs for each scale
        """
        outputs = []
        
        for i, discriminator in enumerate(self.discriminators):
            if i == 0:
                # Original scale
                scale_input = audio
            else:
                # Downsampled scales
                scale_input = self.pooling[i-1](audio)
            
            scale_output = discriminator(scale_input)
            outputs.append(scale_output)
        
        return outputs


class ScaleDiscriminator(nn.Module):
    """
    Single-scale discriminator with grouped convolutions
    """
    
    def __init__(
        self,
        channels: int = 2,
        base_channels: int = 64,
        max_channels: int = 512,
        kernel_sizes: List[int] = [15, 41, 5, 3],
        groups: List[int] = [1, 4, 16, 64],
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # Build discriminator layers
        layers = []
        in_channels = channels
        out_channels = base_channels
        
        for i, (kernel_size, group) in enumerate(zip(kernel_sizes, groups)):
            # Adjust groups based on channel count
            actual_groups = min(group, in_channels, out_channels)
            
            conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2 if i < len(kernel_sizes) - 1 else 1,
                padding=kernel_size // 2,
                groups=actual_groups
            )
            
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            in_channels = out_channels
            out_channels = min(out_channels * 2, max_channels)
        
        # Final classification layer
        final_conv = nn.Conv1d(in_channels, 1, kernel_size=3, padding=1)
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)
        
        layers.append(final_conv)
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through discriminator
        
        Args:
            x: Input audio [batch, channels, time]
            
        Returns:
            Dictionary containing discriminator outputs and feature maps
        """
        feature_maps = []
        
        for layer in self.layers[:-1]:  # All layers except final
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                feature_maps.append(x)
        
        # Final classification
        logits = self.layers[-1](x)
        
        return {
            'logits': logits,
            'feature_maps': feature_maps
        }


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for generator training
    
    Combines adversarial loss with feature matching loss for stable training.
    """
    
    def __init__(
        self,
        loss_type: str = 'hinge',
        feature_matching_weight: float = 10.0,
        use_feature_matching: bool = True
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.feature_matching_weight = feature_matching_weight
        self.use_feature_matching = use_feature_matching
        
        assert loss_type in ['hinge', 'lsgan', 'vanilla'], f"Unknown loss type: {loss_type}"
    
    def generator_loss(self, discriminator_outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute generator adversarial loss
        
        Args:
            discriminator_outputs: List of discriminator outputs for fake samples
            
        Returns:
            Generator adversarial loss
        """
        total_loss = 0.0
        
        for disc_output in discriminator_outputs:
            logits = disc_output['logits']
            
            if self.loss_type == 'hinge':
                loss = -torch.mean(logits)
            elif self.loss_type == 'lsgan':
                loss = torch.mean((logits - 1) ** 2)
            elif self.loss_type == 'vanilla':
                loss = F.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits)
                )
            
            total_loss += loss
        
        return total_loss / len(discriminator_outputs)
    
    def discriminator_loss(
        self,
        real_outputs: List[Dict[str, torch.Tensor]],
        fake_outputs: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute discriminator loss
        
        Args:
            real_outputs: List of discriminator outputs for real samples
            fake_outputs: List of discriminator outputs for fake samples
            
        Returns:
            Discriminator loss
        """
        total_loss = 0.0
        
        for real_output, fake_output in zip(real_outputs, fake_outputs):
            real_logits = real_output['logits']
            fake_logits = fake_output['logits']
            
            if self.loss_type == 'hinge':
                real_loss = torch.mean(F.relu(1 - real_logits))
                fake_loss = torch.mean(F.relu(1 + fake_logits))
            elif self.loss_type == 'lsgan':
                real_loss = torch.mean((real_logits - 1) ** 2)
                fake_loss = torch.mean(fake_logits ** 2)
            elif self.loss_type == 'vanilla':
                real_loss = F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_logits, torch.zeros_like(fake_logits)
                )
            
            total_loss += (real_loss + fake_loss)
        
        return total_loss / len(real_outputs)
    
    def feature_matching_loss(
        self,
        real_outputs: List[Dict[str, torch.Tensor]],
        fake_outputs: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute feature matching loss
        
        Args:
            real_outputs: List of discriminator outputs for real samples
            fake_outputs: List of discriminator outputs for fake samples
            
        Returns:
            Feature matching loss
        """
        total_loss = 0.0
        num_features = 0
        
        for real_output, fake_output in zip(real_outputs, fake_outputs):
            real_features = real_output['feature_maps']
            fake_features = fake_output['feature_maps']
            
            for real_feat, fake_feat in zip(real_features, fake_features):
                loss = F.l1_loss(fake_feat, real_feat.detach())
                total_loss += loss
                num_features += 1
        
        return total_loss / num_features if num_features > 0 else torch.tensor(0.0)
    
    def forward(self, discriminator_outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute generator loss (for backward compatibility)
        
        Args:
            discriminator_outputs: List of discriminator outputs for fake samples
            
        Returns:
            Generator loss
        """
        return self.generator_loss(discriminator_outputs)


class SpectralDiscriminator(nn.Module):
    """
    Discriminator operating in the spectral domain
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        channels: int = 2,
        base_channels: int = 32
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Window for STFT
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)
        
        # Spectral discriminator network
        freq_bins = n_fft // 2 + 1
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=(7, 7), padding=(3, 3)),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(base_channels * 8, 1, kernel_size=(3, 3), padding=(1, 1))
        )
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through spectral discriminator
        
        Args:
            audio: Input audio [batch, channels, time]
            
        Returns:
            Dictionary containing discriminator outputs
        """
        batch_size, channels, time_steps = audio.shape
        
        # Compute STFT for each channel
        spectrograms = []
        for c in range(channels):
            stft = torch.stft(
                audio[:, c, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            spectrograms.append(magnitude)
        
        # Stack spectrograms
        spectrogram = torch.stack(spectrograms, dim=1)  # [batch, channels, freq, time]
        
        # Apply discriminator
        logits = self.conv_layers(spectrogram)
        
        return {
            'logits': logits,
            'feature_maps': []  # Could extract intermediate features if needed
        }


class HybridDiscriminator(nn.Module):
    """
    Hybrid discriminator combining time-domain and spectral discriminators
    """
    
    def __init__(
        self,
        channels: int = 2,
        time_scales: List[int] = [1, 2, 4],
        use_spectral: bool = True,
        spectral_weight: float = 0.5
    ):
        super().__init__()
        
        self.use_spectral = use_spectral
        self.spectral_weight = spectral_weight
        
        # Time-domain multi-scale discriminator
        self.time_discriminator = MultiScaleDiscriminator(
            scales=time_scales,
            channels=channels
        )
        
        # Spectral discriminator
        if use_spectral:
            self.spectral_discriminator = SpectralDiscriminator(
                channels=channels
            )
    
    def forward(self, audio: torch.Tensor) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass through hybrid discriminator
        
        Args:
            audio: Input audio [batch, channels, time]
            
        Returns:
            Dictionary containing time and spectral discriminator outputs
        """
        outputs = {}
        
        # Time-domain discrimination
        outputs['time'] = self.time_discriminator(audio)
        
        # Spectral discrimination
        if self.use_spectral:
            spectral_output = self.spectral_discriminator(audio)
            outputs['spectral'] = [spectral_output]  # Wrap in list for consistency
        
        return outputs 