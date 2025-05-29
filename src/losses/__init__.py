"""
Loss Functions for Neural Audio Codec

This module provides various loss functions for training neural audio codecs,
including reconstruction, perceptual, music-specific, and adversarial losses.
"""

from .reconstruction import ReconstructionLoss
from .perceptual import SpectralLoss, MelSpectralLoss, PerceptualLoss, MFCCLoss
from .music import HarmonicLoss, RhythmicLoss, TimbrePreservationLoss, MusicLoss
from .adversarial import (
    MultiScaleDiscriminator,
    ScaleDiscriminator,
    AdversarialLoss,
    SpectralDiscriminator,
    HybridDiscriminator
)

__all__ = [
    # Reconstruction losses
    'ReconstructionLoss',
    
    # Perceptual losses
    'SpectralLoss',
    'MelSpectralLoss',
    'PerceptualLoss',
    'MFCCLoss',
    
    # Music-specific losses
    'HarmonicLoss',
    'RhythmicLoss',
    'TimbrePreservationLoss',
    'MusicLoss',
    
    # Adversarial components
    'MultiScaleDiscriminator',
    'ScaleDiscriminator',
    'AdversarialLoss',
    'SpectralDiscriminator',
    'HybridDiscriminator'
] 