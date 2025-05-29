#!/usr/bin/env python3
"""
Neural Audio Codec Demo Launcher
================================

Simple script to launch the interactive demo with proper setup.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'gradio',
        'torch',
        'torchaudio', 
        'matplotlib',
        'scipy',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_demo_requirements():
    """Install demo-specific requirements"""
    demo_requirements = [
        'gradio>=4.0.0',
        'matplotlib>=3.5.0',
        'scipy>=1.9.0'
    ]
    
    print("📦 Installing demo requirements...")
    for req in demo_requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False
    
    print("✅ Demo requirements installed!")
    return True

def main():
    """Main launcher function"""
    print("🎵 Neural Audio Codec Demo Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("src/models/codec.py").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where src/ folder is located)")
        sys.exit(1)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"📦 Missing packages: {', '.join(missing)}")
        print("🔧 Installing missing packages...")
        
        if not install_demo_requirements():
            print("❌ Failed to install requirements. Please install manually:")
            print("   pip install gradio matplotlib scipy")
            sys.exit(1)
    
    # Launch demo
    print("🚀 Launching demo...")
    print("📱 The demo will open in your browser automatically")
    print("🌐 A public link will also be created for sharing")
    print("\n" + "=" * 40)
    
    try:
        # Import and run demo
        demo_path = Path("demo/gradio_demo.py")
        if demo_path.exists():
            # Run the demo script
            subprocess.run([sys.executable, str(demo_path)])
        else:
            print("❌ Demo file not found. Creating it...")
            # Create demo directory if it doesn't exist
            demo_path.parent.mkdir(exist_ok=True)
            print("✅ Please run the demo script directly:")
            print("   python demo/gradio_demo.py")
            
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Install requirements: pip install gradio matplotlib scipy")
        print("3. Run directly: python demo/gradio_demo.py")

if __name__ == "__main__":
    main() 