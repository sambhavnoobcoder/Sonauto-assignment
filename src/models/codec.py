"""
Neural Audio Codec - Main Model

This module implements the complete neural audio codec architecture,
combining hierarchical encoding, vector quantization, and decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .encoder import HierarchicalEncoder
from .decoder import HierarchicalDecoder
from .quantizer import MusicVectorQuantizer


class NeuralAudioCodec(nn.Module):
    """
    Advanced Neural Audio Codec for Music
    
    Features:
    - Hierarchical multi-scale encoding/decoding
    - Music-optimized vector quantization
    - Perceptual loss integration
    - Variable bitrate support
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        n_fft: int = 2048,
        hop_length: int = 512,
        encoder_dim: int = 128,
        decoder_dim: int = 128,
        codebook_size: int = 256,
        codebook_dim: int = 256,
        num_quantizers: int = 2,
        commitment_weight: float = 0.25,
        compression_ratios: List[int] = [8, 16, 32],
        use_music_features: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.compression_ratios = compression_ratios
        self.use_music_features = use_music_features
        
        # Core components
        self.encoder = HierarchicalEncoder(
            channels=channels,
            encoder_dim=encoder_dim,
            compression_ratios=compression_ratios,
            use_music_features=use_music_features
        )
        
        self.quantizer = MusicVectorQuantizer(
            codebook_size=1024,
            codebook_dim=256,
            num_quantizers=8,
            commitment_weight=0.25,
            encoder_dim=encoder_dim
        )
        
        self.decoder = HierarchicalDecoder(
            decoder_dim=decoder_dim,
            channels=channels,
            compression_ratios=compression_ratios,
            use_music_features=use_music_features,
            encoder_dim=encoder_dim
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Simple pre-training flag
        self._is_pretrained = False
    
    def _initialize_weights(self):
        """Apply better weight initialization for audio processing"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                # Use smaller initialization for audio to prevent saturation
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Smaller initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                # Standard initialization for normalization layers
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode audio to hierarchical representations
        
        Args:
            audio: Input audio tensor [batch, channels, time]
            
        Returns:
            Dictionary containing encoded representations and metadata
        """
        # Hierarchical encoding
        encoded = self.encoder(audio)
        
        # Vector quantization
        quantized_outputs = []
        total_commitment_loss = 0
        total_codebook_loss = 0
        
        for i, enc_level in enumerate(encoded['hierarchical_features']):
            quant_out = self.quantizer(enc_level, level=i)
            quantized_outputs.append(quant_out)
            total_commitment_loss += quant_out['commitment_loss']
            total_codebook_loss += quant_out['codebook_loss']
        
        return {
            'quantized_features': [q['quantized'] for q in quantized_outputs],
            'codes': [q['codes'] for q in quantized_outputs],
            'commitment_loss': total_commitment_loss,
            'codebook_loss': total_codebook_loss,
            'music_features': encoded.get('music_features', None),
            'perceptual_features': encoded.get('perceptual_features', None)
        }
    
    def decode(self, encoded_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decode quantized representations back to audio
        
        Args:
            encoded_data: Dictionary containing quantized features and metadata
            
        Returns:
            Reconstructed audio tensor [batch, channels, time]
        """
        return self.decoder(
            quantized_features=encoded_data['quantized_features'],
            music_features=encoded_data.get('music_features', None),
            perceptual_features=encoded_data.get('perceptual_features', None)
        )
    
    def _match_size(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """Helper method to match tensor size to target size"""
        current_size = tensor.shape[-1]
        
        if current_size > target_size:
            # Trim tensor to match target
            return tensor[..., :target_size]
        elif current_size < target_size:
            # Pad tensor to match target
            pad_size = target_size - current_size
            return F.pad(tensor, (0, pad_size), mode='reflect')
        else:
            return tensor

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode
        
        Args:
            audio: Input audio tensor [batch, channels, time]
            
        Returns:
            Dictionary containing reconstructed audio and losses
        """
        original_size = audio.shape[-1]
        
        # Encode
        encoded = self.encode(audio)
        
        # Decode
        reconstructed = self.decode(encoded)
        
        # Ensure reconstructed audio matches original size
        reconstructed = self._match_size(reconstructed, original_size)
        
        return {
            'audio': reconstructed,  # For trainer compatibility
            'reconstructed': reconstructed,  # For test script compatibility
            'codes': encoded['codes'],
            'quantized': encoded['quantized_features'],  # For trainer compatibility
            'commitment_loss': encoded['commitment_loss'],
            'codebook_loss': encoded['codebook_loss'],
            'music_features': encoded.get('music_features', None),
            'perceptual_features': encoded.get('perceptual_features', None)
        }
    
    def compress(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress audio to codes for storage/transmission
        
        Args:
            audio: Input audio tensor [batch, channels, time]
            
        Returns:
            Dictionary containing compressed codes and metadata
        """
        encoded = self.encode(audio)
        
        return {
            'codes': encoded['codes'],
            'shape': audio.shape,
            'sample_rate': self.sample_rate,
            'compression_ratios': self.compression_ratios
        }
    
    def decompress(self, compressed_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decompress codes back to audio
        
        Args:
            compressed_data: Dictionary containing codes and metadata
            
        Returns:
            Reconstructed audio tensor
        """
        # Reconstruct quantized features from codes
        quantized_features = []
        for i, codes in enumerate(compressed_data['codes']):
            quantized = self.quantizer.decode_codes(codes, level=i)
            quantized_features.append(quantized)
        
        # Decode to audio
        return self.decoder(
            quantized_features=quantized_features,
            music_features=None,  # Not stored in compressed format
            perceptual_features=None
        )
    
    def get_compression_ratio(self, audio_length: int) -> float:
        """
        Calculate the compression ratio for given audio length
        
        Args:
            audio_length: Length of input audio in samples
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        # Calculate total compression across all levels
        total_compression = 1
        for ratio in self.compression_ratios:
            total_compression *= ratio
        
        # Account for quantization (bits per code)
        bits_per_code = np.log2(self.quantizer.codebook_size)
        compressed_length = audio_length // total_compression
        
        # Original: 16-bit stereo, Compressed: bits_per_code * num_quantizers
        original_bits = audio_length * self.channels * 16
        compressed_bits = compressed_length * self.quantizer.num_quantizers * bits_per_code
        
        return original_bits / compressed_bits
    
    def get_model_info(self) -> Dict:
        """Get model configuration and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'NeuralAudioCodec',
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'compression_ratios': self.compression_ratios,
            'codebook_size': self.quantizer.codebook_size,
            'num_quantizers': self.quantizer.num_quantizers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_music_features': self.use_music_features
        }
    
    def comprehensive_pretrain(self, audio_tensor, num_epochs=8):
        """Enhanced comprehensive pre-training with advanced techniques"""
        print("🚀 Starting enhanced comprehensive pre-training...")
        self.train()
        
        # Progressive learning rates - start higher, decay more gradually
        base_lr = 5e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=1e-5, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
        
        # Create more diverse training data
        batch_size, channels, length = audio_tensor.shape
        training_data = []
        
        # Original audio (multiple copies for emphasis)
        for _ in range(3):
            training_data.append(audio_tensor)
        
        # Time-shifted versions with different shifts
        for shift in [500, 1000, 2000, 4000, 8000]:
            if length > shift:
                shifted = torch.cat([audio_tensor[..., shift:], audio_tensor[..., :shift]], dim=-1)
                training_data.append(shifted)
        
        # Amplitude variations (more conservative)
        for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            scaled = audio_tensor * scale
            scaled = torch.clamp(scaled, -1.0, 1.0)
            training_data.append(scaled)
        
        # Frequency filtering (simulate different audio conditions)
        for cutoff in [0.8, 0.9, 1.0]:  # High-pass effect
            filtered = audio_tensor * cutoff + audio_tensor * (1 - cutoff) * 0.1
            training_data.append(filtered)
        
        # Small noise additions (very conservative)
        for noise_level in [0.005, 0.01, 0.015]:
            noisy = audio_tensor + torch.randn_like(audio_tensor) * noise_level
            noisy = torch.clamp(noisy, -1.0, 1.0)
            training_data.append(noisy)
        
        print(f"Created {len(training_data)} diverse training samples")
        
        # Progressive training with increasing complexity
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_samples = 0
            
            # Progressive loss weights - start simple, add complexity
            time_weight = 1.0
            freq_weight = 0.3 + (epoch / num_epochs) * 0.7  # Gradually increase
            phase_weight = 0.1 + (epoch / num_epochs) * 0.3
            perceptual_weight = 0.2 + (epoch / num_epochs) * 0.5
            commitment_weight = 0.005 + (epoch / num_epochs) * 0.015
            
            for i, target in enumerate(training_data):
                optimizer.zero_grad()
                
                # All training samples should now be the same length as original
                # Just ensure they match exactly
                if target.shape[-1] != length:
                    if target.shape[-1] > length:
                        target = target[..., :length]
                    else:
                        # This shouldn't happen now, but just in case
                        pad_size = length - target.shape[-1]
                        target = F.pad(target, (0, pad_size), mode='reflect')
                
                # Forward pass
                result = self(target)
                reconstructed = result['audio']
                
                # Ensure size match
                reconstructed = self._match_size(reconstructed, target.shape[-1])
                
                # Enhanced multi-component loss
                # 1. Time domain reconstruction (L1 + L2 combination)
                time_loss_l2 = F.mse_loss(reconstructed, target)
                time_loss_l1 = F.l1_loss(reconstructed, target)
                time_loss = 0.7 * time_loss_l2 + 0.3 * time_loss_l1
                
                # 2. Multi-scale frequency domain reconstruction
                freq_loss = 0
                for n_fft in [512, 1024, 2048]:
                    if target.shape[-1] >= n_fft:
                        target_fft = torch.fft.rfft(target, n=n_fft, dim=-1)
                        recon_fft = torch.fft.rfft(reconstructed, n=n_fft, dim=-1)
                        
                        # Magnitude loss
                        mag_loss = F.mse_loss(torch.abs(target_fft), torch.abs(recon_fft))
                        freq_loss += mag_loss
                        
                        # Log magnitude loss for better perceptual quality
                        log_mag_loss = F.mse_loss(
                            torch.log(torch.abs(target_fft) + 1e-7),
                            torch.log(torch.abs(recon_fft) + 1e-7)
                        )
                        freq_loss += 0.5 * log_mag_loss
                
                # 3. Enhanced phase preservation
                target_fft = torch.fft.rfft(target, dim=-1)
                recon_fft = torch.fft.rfft(reconstructed, dim=-1)
                
                # Phase difference loss
                target_phase = torch.angle(target_fft)
                recon_phase = torch.angle(recon_fft)
                phase_diff = torch.abs(target_phase - recon_phase)
                phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)  # Wrap around
                phase_loss = torch.mean(phase_diff)
                
                # 4. Multi-scale perceptual loss
                perceptual_loss = 0
                for scale in [1, 2, 4, 8]:
                    if target.shape[-1] >= scale * 2:
                        target_down = F.avg_pool1d(target, kernel_size=scale, stride=scale)
                        recon_down = F.avg_pool1d(reconstructed, kernel_size=scale, stride=scale)
                        perceptual_loss += F.mse_loss(target_down, recon_down)
                        
                        # Add spectral loss at each scale
                        if target_down.shape[-1] >= 256:
                            target_spec = torch.abs(torch.fft.rfft(target_down, dim=-1))
                            recon_spec = torch.abs(torch.fft.rfft(recon_down, dim=-1))
                            perceptual_loss += 0.3 * F.mse_loss(target_spec, recon_spec)
                
                # 5. Commitment loss with adaptive weight
                commitment_loss = result.get('commitment_loss', 0)
                
                # 6. Feature matching loss (encourage diverse codebook usage)
                feature_loss = 0
                if 'quantized' in result:
                    for quant_feat in result['quantized']:
                        # Encourage feature diversity
                        feat_var = torch.var(quant_feat, dim=-1).mean()
                        feature_loss += torch.exp(-feat_var)  # Penalty for low variance
                
                # Combined loss with progressive weights
                total_loss = (time_weight * time_loss + 
                             freq_weight * freq_loss + 
                             phase_weight * phase_loss + 
                             perceptual_weight * perceptual_loss + 
                             commitment_weight * commitment_loss +
                             0.1 * feature_loss)
                
                # Backward pass with gradient scaling
                total_loss.backward()
                
                # Adaptive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += total_loss.item()
                num_samples += 1
                
                if i % 5 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1}/{num_epochs}, Sample {i+1}/{len(training_data)}")
                    print(f"    Loss: {total_loss.item():.4f}, LR: {current_lr:.6f}, Grad: {grad_norm:.3f}")
            
            avg_loss = epoch_loss / num_samples
            print(f"✅ Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
            
            # Validation check every few epochs
            if (epoch + 1) % 2 == 0:
                self.eval()
                with torch.no_grad():
                    test_result = self(audio_tensor)
                    test_recon = self._match_size(test_result['audio'], audio_tensor.shape[-1])
                    test_mse = F.mse_loss(test_recon, audio_tensor).item()
                    
                    # Calculate SNR for progress tracking
                    signal_power = torch.mean(audio_tensor ** 2)
                    noise_power = torch.mean((audio_tensor - test_recon) ** 2)
                    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
                    
                    print(f"📊 Validation - MSE: {test_mse:.6f}, SNR: {snr:.2f} dB")
                self.train()
        
        self.eval()
        self._is_pretrained = True
        print("🎉 Enhanced comprehensive pre-training completed!")
        
        # Final evaluation
        with torch.no_grad():
            final_result = self(audio_tensor)
            final_recon = self._match_size(final_result['audio'], audio_tensor.shape[-1])
            final_mse = F.mse_loss(final_recon, audio_tensor).item()
            
            signal_power = torch.mean(audio_tensor ** 2)
            noise_power = torch.mean((audio_tensor - final_recon) ** 2)
            final_snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
            
            print(f"🏆 Final Results - MSE: {final_mse:.6f}, SNR: {final_snr:.2f} dB")
    
    def quick_pretrain(self, audio_tensor, num_steps=50):
        """Quick pre-training - now calls comprehensive pre-training"""
        if self._is_pretrained:
            return
        
        # Use comprehensive pre-training instead
        self.comprehensive_pretrain(audio_tensor, num_epochs=3) 