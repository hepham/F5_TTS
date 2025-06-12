#!/usr/bin/env python3
"""
Simple script to run multilingual tests for F5-TTS Django API
Tests each supported language with one sentence
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Check if audio file is provided
    if len(sys.argv) < 2:
        print("Usage: python run_multilingual_test.py <reference_audio.wav>")
        print("Example: python run_multilingual_test.py my_voice.wav")
        return
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        return
    
    print("🌍 Starting multilingual test for F5-TTS Django API")
    print(f"📁 Using reference audio: {audio_file}")
    
    # Run the multilingual test
    try:
        result = subprocess.run([
            "python", "test_django_api.py",
            "--audio", audio_file,
            "--multilingual"
        ], check=True)
        
        print("\n✅ Multilingual test completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error code: {e.returncode}")
        
    except FileNotFoundError:
        print("❌ Error: test_django_api.py not found in current directory")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")

if __name__ == "__main__":
    main() 