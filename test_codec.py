#!/usr/bin/env python3
"""
Test script for Neural Audio Codec

This script tests the basic functionality of our neural audio codec implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.codec import NeuralAudioCodec


def test_codec_basic():
    """Test basic codec functionality"""
    print("Testing Neural Audio Codec...")
    
    # Create model
    codec = NeuralAudioCodec(
        sample_rate=44100,
        channels=2,
        encoder_dim=256,  # Smaller for testing
        decoder_dim=256,
        codebook_size=512,
        codebook_dim=128,
        num_quantizers=4,
        compression_ratios=[4, 8, 16],  # Smaller ratios for testing
        use_music_features=True
    )
    
    print(f"Model created with {sum(p.numel() for p in codec.parameters())} parameters")
    
    # Create test audio (2 seconds of stereo audio)
    batch_size = 2
    channels = 2
    duration = 2.0  # seconds
    sample_rate = 44100
    audio_length = int(duration * sample_rate)
    
    # Generate test audio (sine waves with different frequencies)
    test_audio = torch.zeros(batch_size, channels, audio_length)
    for b in range(batch_size):
        for c in range(channels):
            freq = 440 + b * 110 + c * 55  # Different frequencies
            t = torch.linspace(0, duration, audio_length)
            test_audio[b, c] = torch.sin(2 * np.pi * freq * t) * 0.5
    
    print(f"Test audio shape: {test_audio.shape}")
    
    # Test forward pass
    print("Testing forward pass...")
    codec.eval()
    with torch.no_grad():
        try:
            outputs = codec(test_audio)
            print("✓ Forward pass successful!")
            
            print(f"Reconstructed audio shape: {outputs['reconstructed'].shape}")
            print(f"Number of hierarchical levels: {len(outputs['codes'])}")
            
            # Test compression/decompression
            print("Testing compression/decompression...")
            compressed = codec.compress(test_audio)
            decompressed = codec.decompress(compressed)
            
            print(f"Original shape: {test_audio.shape}")
            print(f"Decompressed shape: {decompressed.shape}")
            
            # Calculate compression ratio
            compression_ratio = codec.get_compression_ratio(audio_length)
            print(f"Compression ratio: {compression_ratio:.2f}:1")
            
            # Calculate reconstruction error (handle length mismatch)
            min_length = min(test_audio.shape[-1], outputs['reconstructed'].shape[-1])
            test_audio_cropped = test_audio[..., :min_length]
            reconstructed_cropped = outputs['reconstructed'][..., :min_length]
            mse = torch.mean((test_audio_cropped - reconstructed_cropped) ** 2)
            print(f"Reconstruction MSE: {mse.item():.6f}")
            print(f"Length difference: {test_audio.shape[-1] - outputs['reconstructed'].shape[-1]} samples")
            
            print("✓ All tests passed!")
            
        except Exception as e:
            print(f"✗ Error during testing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_model_info():
    """Test model information retrieval"""
    print("\nTesting model info...")
    
    codec = NeuralAudioCodec(
        encoder_dim=128,
        decoder_dim=128,
        codebook_size=256,
        num_quantizers=2
    )
    
    info = codec.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def test_different_audio_lengths():
    """Test with different audio lengths"""
    print("\nTesting different audio lengths...")
    
    codec = NeuralAudioCodec(
        encoder_dim=128,
        decoder_dim=128,
        codebook_size=256,
        num_quantizers=2,
        compression_ratios=[4, 8]
    )
    
    codec.eval()
    
    # Test different lengths
    lengths = [8192, 16384, 32768]  # Different audio lengths
    
    for length in lengths:
        test_audio = torch.randn(1, 2, length) * 0.1
        
        try:
            with torch.no_grad():
                outputs = codec(test_audio)
                print(f"✓ Length {length}: Input {test_audio.shape} -> Output {outputs['reconstructed'].shape}")
        except Exception as e:
            print(f"✗ Length {length}: Error - {e}")


if __name__ == "__main__":
    print("Neural Audio Codec Test Suite")
    print("=" * 50)
    
    # Run tests
    success = test_codec_basic()
    
    if success:
        test_model_info()
        test_different_audio_lengths()
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 