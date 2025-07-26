#!/usr/bin/env python3
"""
Setup script for Quiz Craft Backend - Phase 2
"""
import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is 3.7 or higher"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call(["py", "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def setup_environment():
    """Setup environment file"""
    env_file = ".env"
    env_example = ".env.example"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            print("📝 Creating .env file from template...")
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created")
            print("⚠️  Please edit .env file and add your MISTRAL_API_KEY")
        else:
            print("⚠️  .env.example not found, creating basic .env file...")
            with open(env_file, 'w') as f:
                f.write("MISTRAL_API_KEY=your_mistral_api_key_here\n")
            print("✅ Basic .env file created")
            print("⚠️  Please edit .env file and add your MISTRAL_API_KEY")
    else:
        print("✅ .env file already exists")

def main():
    print("Quiz Craft Backend Setup - Phase 2")
    print("=" * 40)
    
    check_python_version()
    install_requirements()
    setup_environment()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file and add your MISTRAL_API_KEY")
    print("2. Run: py server.py")
    print("3. Test with: py test_ocr.py")

if __name__ == "__main__":
    main()
