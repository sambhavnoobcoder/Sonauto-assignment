#!/usr/bin/env python3
"""
Pretrain Neural Audio Codec
==========================

This script pretrains the neural audio codec model on a single audio file
for quick inference setup.
"""

import torch
import torchaudio
import json
from pathlib import Path
from src.models.codec import NeuralAudioCodec
from src.training.trainer import AudioCodecTrainer, create_trainer_from_config

def load_audio(audio_path: str, target_sr: int = 44100):
    """Load and preprocess audio file"""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Ensure shape is [batch, channels, time]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    
    return waveform

def main():
    # Configuration
    config = {
        "sample_rate": 44100,
        "channels": 2,
        "encoder_dim": 128,
        "decoder_dim": 128,
        "codebook_size": 1024,
        "codebook_dim": 256,
        "num_quantizers": 8,
        "commitment_weight": 0.25,
        "compression_ratios": [8, 16, 32],
        "use_music_features": True,
        "losses": {
            "reconstruction_weight": 1.0,
            "perceptual_weight": 1.0,
            "music_weight": 1.0,
            "use_perceptual": True,
            "use_music": True
        },
        "optimizer": {
            "gen_lr": 1e-4,
            "disc_lr": 1e-4,
            "betas": [0.8, 0.99],
            "weight_decay": 1e-4
        }
    }
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralAudioCodec(**config)
    model.to(device)
    
    # Load audio
    audio_path = "jubin-nautiyal--ami-mishra--parth-s-diksha-s--kunaal-v-ashish-p-bhushan-kumar.wav"
    audio = load_audio(audio_path)
    audio = audio.to(device)
    
    print(f"Loaded audio shape: {audio.shape}")
    
    # Create experiment directory
    experiment_dir = Path("experiments/pretrained")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = AudioCodecTrainer(
        model=model,
        config=config,
        device=device,
        experiment_dir=str(experiment_dir)
    )
    
    # Comprehensive pretraining
    print("Starting comprehensive pretraining...")
    model.comprehensive_pretrain(audio, num_epochs=8)
    
    # Save pretrained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, experiment_dir / "pretrained_model.pt")
    
    print("Pretraining complete! Model saved to experiments/pretrained/pretrained_model.pt")

if __name__ == "__main__":
    main() 