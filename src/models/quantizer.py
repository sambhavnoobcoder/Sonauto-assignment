"""
Music Vector Quantizer for Neural Audio Codec

This module implements a sophisticated vector quantization scheme optimized for music,
featuring multiple codebooks, commitment losses, and music-aware quantization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with exponential moving average updates
    """
    
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(codebook_size, codebook_dim))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', self.codebook.clone())
        
        # Normalization
        self.codebook.data.uniform_(-1/codebook_size, 1/codebook_size)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Quantize input features
        
        Args:
            x: Input features [batch, dim, time]
            
        Returns:
            Dictionary containing quantized features and losses
        """
        # Flatten spatial dimensions
        batch_size, dim, time = x.shape
        x_flat = x.permute(0, 2, 1).contiguous().view(-1, dim)  # [batch*time, dim]
        
        # Calculate distances to codebook entries
        distances = torch.cdist(x_flat, self.codebook)  # [batch*time, codebook_size]
        
        # Find closest codebook entries
        codes = torch.argmin(distances, dim=1)  # [batch*time]
        
        # Get quantized features
        quantized_flat = F.embedding(codes, self.codebook)  # [batch*time, dim]
        quantized = quantized_flat.view(batch_size, time, dim).permute(0, 2, 1)  # [batch, dim, time]
        
        # Calculate losses
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Update codebook (only during training)
        if self.training:
            self._update_codebook(x_flat, codes)
        
        return {
            'quantized': quantized,
            'codes': codes.view(batch_size, time),
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'perplexity': self._calculate_perplexity(codes)
        }
    
    def _update_codebook(self, x_flat: torch.Tensor, codes: torch.Tensor):
        """Update codebook using exponential moving average"""
        
        # Calculate cluster assignments
        encodings = F.one_hot(codes, self.codebook_size).float()  # [batch*time, codebook_size]
        
        # Update cluster sizes
        cluster_size = encodings.sum(0)
        self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        
        # Update embeddings
        embed_sum = encodings.t() @ x_flat  # [codebook_size, dim]
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        
        # Normalize embeddings
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.codebook.data.copy_(embed_normalized)
    
    def _calculate_perplexity(self, codes: torch.Tensor) -> torch.Tensor:
        """Calculate perplexity of code usage"""
        encodings = F.one_hot(codes, self.codebook_size).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity
    
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes back to features"""
        return F.embedding(codes, self.codebook).permute(0, 2, 1)


class MusicAwareQuantizer(VectorQuantizer):
    """
    Music-aware vector quantizer that adapts to musical characteristics
    """
    
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_weight: float = 0.25,
        music_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(codebook_size, codebook_dim, commitment_weight, **kwargs)
        
        self.music_weight = music_weight
        
        # Music-specific codebook organization
        # Organize codebook into semantic clusters for different musical elements
        self.harmonic_indices = torch.arange(0, codebook_size // 3)
        self.rhythmic_indices = torch.arange(codebook_size // 3, 2 * codebook_size // 3)
        self.timbral_indices = torch.arange(2 * codebook_size // 3, codebook_size)
        
        # Learnable music feature projector
        self.music_projector = nn.Linear(codebook_dim, 3)  # Project to 3 music aspects
        
    def forward(self, x: torch.Tensor, music_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Music-aware quantization
        
        Args:
            x: Input features [batch, dim, time]
            music_context: Optional music context features [batch, music_dim, time]
            
        Returns:
            Dictionary containing quantized features and losses
        """
        # Standard quantization
        result = super().forward(x)
        
        # Add music-aware bias if context is provided
        if music_context is not None:
            music_loss = self._compute_music_loss(x, music_context, result['codes'])
            result['music_loss'] = music_loss
            result['commitment_loss'] = result['commitment_loss'] + self.music_weight * music_loss
        
        return result
    
    def _compute_music_loss(
        self, 
        x: torch.Tensor, 
        music_context: torch.Tensor, 
        codes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute music-aware loss that encourages semantically meaningful quantization
        """
        batch_size, dim, time = x.shape
        
        # Project features to music space
        x_flat = x.permute(0, 2, 1).contiguous().view(-1, dim)
        music_proj = self.music_projector(x_flat)  # [batch*time, 3]
        music_proj = music_proj.view(batch_size, time, 3).permute(0, 2, 1)  # [batch, 3, time]
        
        # Calculate music similarity
        music_sim = F.cosine_similarity(music_proj, music_context, dim=1)  # [batch, time]
        
        # Encourage codes to respect music similarity
        codes_flat = codes.view(-1)
        
        # Calculate code consistency loss
        consistency_loss = 0
        for i in range(time - 1):
            current_codes = codes[:, i]
            next_codes = codes[:, i + 1]
            current_sim = music_sim[:, i]
            next_sim = music_sim[:, i + 1]
            
            # If music similarity is high, codes should be similar
            sim_diff = torch.abs(current_sim - next_sim)
            code_diff = (current_codes != next_codes).float()
            
            consistency_loss += torch.mean(sim_diff * code_diff)
        
        return consistency_loss / (time - 1)


class MusicVectorQuantizer(nn.Module):
    """
    Multi-level vector quantizer optimized for music
    
    Features:
    - Multiple quantizers for different hierarchical levels
    - Music-aware quantization strategies
    - Adaptive commitment weights
    """
    
    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        num_quantizers: int = 8,
        commitment_weight: float = 0.25,
        music_weight: float = 0.1,
        use_music_aware: bool = True,
        encoder_dim: int = None  # Add encoder_dim parameter
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight
        self.use_music_aware = use_music_aware
        
        # Create multiple quantizers
        self.quantizers = nn.ModuleList()
        
        for i in range(num_quantizers):
            if use_music_aware:
                quantizer = MusicAwareQuantizer(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    commitment_weight=commitment_weight,
                    music_weight=music_weight
                )
            else:
                quantizer = VectorQuantizer(
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    commitment_weight=commitment_weight
                )
            
            self.quantizers.append(quantizer)
        
        # Input projection layers for different hierarchical levels
        # These should map from encoder output dims to codebook_dim
        if encoder_dim is None:
            encoder_dim = codebook_dim
        
        # Calculate actual encoder output dimensions based on the encoder_dim
        encoder_dims = [
            encoder_dim // 4,  # Level 0: encoder_dim // 4
            encoder_dim // 2,  # Level 1: encoder_dim // 2  
            encoder_dim        # Level 2: encoder_dim
        ]
        
        self.input_projections = nn.ModuleList([
            nn.Conv1d(encoder_dims[min(i, len(encoder_dims)-1)], codebook_dim, kernel_size=1)
            for i in range(num_quantizers)
        ])
        
        # Reverse projections for decoding (codebook_dim -> encoder_dims)
        self.output_projections = nn.ModuleList([
            nn.Conv1d(codebook_dim, encoder_dims[min(i, len(encoder_dims)-1)], kernel_size=1)
            for i in range(num_quantizers)
        ])
        
        # Adaptive weight network
        self.weight_network = nn.Sequential(
            nn.Conv1d(codebook_dim, codebook_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(codebook_dim // 4, num_quantizers, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Initialize codebooks with better patterns for audio
        self._initialize_codebooks()
    
    def forward(
        self, 
        x: torch.Tensor, 
        level: int = 0,
        music_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-quantizer forward pass
        
        Args:
            x: Input features [batch, dim, time]
            level: Hierarchical level (affects quantizer selection)
            music_context: Optional music context features
            
        Returns:
            Dictionary containing quantized features and losses
        """
        batch_size, dim, time = x.shape
        
        # Project input for this level
        if level < len(self.input_projections):
            x_proj = self.input_projections[level](x)
        else:
            x_proj = x
        
        # Calculate adaptive weights
        weights = self.weight_network(x_proj)  # [batch, num_quantizers, time]
        
        # Apply quantizers
        quantized_outputs = []
        total_commitment_loss = 0
        total_codebook_loss = 0
        total_music_loss = 0
        
        for i, quantizer in enumerate(self.quantizers):
            if self.use_music_aware and music_context is not None:
                quant_out = quantizer(x_proj, music_context)
                if 'music_loss' in quant_out:
                    total_music_loss += quant_out['music_loss']
            else:
                quant_out = quantizer(x_proj)
            
            quantized_outputs.append(quant_out)
            total_commitment_loss += quant_out['commitment_loss']
            total_codebook_loss += quant_out['codebook_loss']
        
        # Weighted combination of quantized features
        quantized_combined = torch.zeros_like(x_proj)
        codes_combined = torch.zeros(batch_size, time, dtype=torch.long, device=x.device)
        
        for i, quant_out in enumerate(quantized_outputs):
            weight = weights[:, i:i+1, :]  # [batch, 1, time]
            quantized_combined += weight * quant_out['quantized']
            
            # Use the quantizer with highest weight for codes
            max_weight_mask = (weights[:, i, :] == weights.max(dim=1)[0])
            codes_combined[max_weight_mask] = quant_out['codes'][max_weight_mask]
        
        result = {
            'quantized': quantized_combined,
            'codes': codes_combined,
            'commitment_loss': total_commitment_loss / self.num_quantizers,
            'codebook_loss': total_codebook_loss / self.num_quantizers,
            'weights': weights
        }
        
        if total_music_loss > 0:
            result['music_loss'] = total_music_loss / self.num_quantizers
        
        return result
    
    def decode_codes(self, codes: torch.Tensor, level: int = 0) -> torch.Tensor:
        """
        Decode codes back to features
        
        Args:
            codes: Quantization codes [batch, time]
            level: Hierarchical level
            
        Returns:
            Decoded features [batch, dim, time]
        """
        # Use the first quantizer for decoding (could be made more sophisticated)
        decoded = self.quantizers[0].decode_codes(codes)
        
        # Apply level-specific reverse projection (codebook_dim -> encoder_dim)
        if level < len(self.output_projections):
            decoded = self.output_projections[level](decoded)
        
        return decoded
    
    def get_codebook_usage(self) -> Dict[str, torch.Tensor]:
        """Get codebook usage statistics"""
        usage_stats = {}
        
        for i, quantizer in enumerate(self.quantizers):
            cluster_size = quantizer.cluster_size
            total_usage = cluster_size.sum()
            usage_ratio = cluster_size / (total_usage + 1e-8)
            
            usage_stats[f'quantizer_{i}'] = {
                'cluster_sizes': cluster_size,
                'usage_ratio': usage_ratio,
                'active_codes': (cluster_size > 0).sum(),
                'entropy': -torch.sum(usage_ratio * torch.log(usage_ratio + 1e-8))
            }
        
        return usage_stats

    def _initialize_codebooks(self):
        """Initialize codebooks with patterns suitable for audio features"""
        for i, quantizer in enumerate(self.quantizers):
            if hasattr(quantizer, 'codebook'):
                # Initialize with smaller variance for audio
                nn.init.normal_(quantizer.codebook, mean=0.0, std=0.02)
                
                # Add some structured patterns for the first few entries
                with torch.no_grad():
                    # Create sinusoidal patterns at different frequencies
                    for j in range(min(32, self.codebook_size)):
                        freq = (j + 1) * 0.1
                        pattern = torch.sin(torch.arange(self.codebook_dim, dtype=torch.float32) * freq)
                        quantizer.codebook[j] = pattern * 0.1 