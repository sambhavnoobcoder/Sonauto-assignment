"""
Neural Audio Codec - Advanced Music-Optimized Audio Compression

A state-of-the-art neural audio codec specifically designed for music generation
and compression, featuring hierarchical tokenization and music-aware perceptual losses.

Author: Your Name
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.codec import NeuralAudioCodec
from .models.encoder import HierarchicalEncoder
from .models.decoder import HierarchicalDecoder
from .models.quantizer import MusicVectorQuantizer

__all__ = [
    "NeuralAudioCodec",
    "HierarchicalEncoder", 
    "HierarchicalDecoder",
    "MusicVectorQuantizer"
] 