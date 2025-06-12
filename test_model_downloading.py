#!/usr/bin/env python3
"""
Test script for model downloading and caching functionality
Demonstrates how the API downloads and caches language-specific models from Hugging Face
"""

import requests
import json
import time
import os
from pathlib import Path

# Test configuration
API_BASE_URL = "http://localhost:8000/api/tts"
MODELS_DIR = "models"

def check_model_cache():
    """Check what models are currently cached locally"""
    models_dir = Path(MODELS_DIR)
    if not models_dir.exists():
        print("🔍 No models directory found yet.")
        return
    
    print("🔍 Checking local model cache...")
    cached_models = []
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "model.safetensors"
            if model_file.exists():
                size = model_file.stat().st_size
                size_mb = size / (1024 * 1024)
                cached_models.append({
                    "name": model_dir.name.replace("_", "/").replace(":", "/"),
                    "size_mb": round(size_mb, 2),
                    "path": str(model_file)
                })
    
    if cached_models:
        print(f"📦 Found {len(cached_models)} cached models:")
        for model in cached_models:
            print(f"   • {model['name']} ({model['size_mb']} MB)")
    else:
        print("📦 No cached models found.")
    
    return cached_models

def test_model_downloading():
    """Test automatic model downloading through the API"""
    
    print("🚀 Testing Model Downloading and Caching System")
    print("=" * 60)
    
    # Check initial cache state
    initial_models = check_model_cache()
    
    # Test cases that will trigger different model downloads
    test_cases = [
        {
            "language": "en",
            "text": "Hello, this will download the English model.",
            "expected_model": "SWivid/F5-TTS_v1",
            "description": "English (should download SWivid/F5-TTS_v1)"
        },
        {
            "language": "es", 
            "text": "Hola, esto descargará el modelo español.",
            "expected_model": "jpgallegoar/F5-Spanish",
            "description": "Spanish (should download jpgallegoar/F5-Spanish)"
        },
        {
            "language": "fr",
            "text": "Bonjour, ceci téléchargera le modèle français.",
            "expected_model": "RASPIAUDIO/F5-French-MixedSpeakers-reduced",
            "description": "French (should download RASPIAUDIO/F5-French-MixedSpeakers-reduced)"
        }
    ]
    
    print(f"\n📋 Testing {len(test_cases)} different language models...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"   Language: {test_case['language']}")
        print(f"   Expected Model: {test_case['expected_model']}")
        
        # Check if this model is already cached
        model_name = test_case['expected_model'].replace("/", "_").replace(":", "_")
        model_path = Path(MODELS_DIR) / model_name / "model.safetensors"
        
        if model_path.exists():
            print(f"   ✅ Model already cached at: {model_path}")
        else:
            print(f"   📥 Model not cached, will trigger download...")
        
        # Make API request
        start_time = time.time()
        
        try:
            # Create a dummy audio file for the request
            audio_data = {
                'text': test_case['text'],
                'language': test_case['language'],
                'model': 'Auto',  # This will trigger language-specific model selection
                'ref_text': 'Sample reference text',
                'remove_silence': False
            }
            
            # Create a small dummy audio file
            import tempfile
            import wave
            import numpy as np
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Create a short sine wave
                sample_rate = 24000
                duration = 1.0  # 1 second
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_signal = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz sine wave
                
                # Save as WAV
                with wave.open(temp_audio.name, 'w') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_signal * 32767).astype(np.int16).tobytes())
                
                # Make the API request
                with open(temp_audio.name, 'rb') as audio_file:
                    files = {'ref_audio': audio_file}
                    
                    print(f"   🔄 Making API request...")
                    response = requests.post(
                        f"{API_BASE_URL}/generate/",
                        files=files,
                        data=audio_data,
                        timeout=300  # 5 minute timeout for model download
                    )
                
                # Cleanup
                os.unlink(temp_audio.name)
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success! (took {elapsed_time:.1f}s)")
                print(f"   📄 Response: {result.get('message', 'Generated successfully')}")
                
                # Check if model was downloaded
                if model_path.exists():
                    size = model_path.stat().st_size / (1024 * 1024)
                    print(f"   💾 Model cached: {model_path} ({size:.1f} MB)")
                
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   📄 Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out (model download may take several minutes)")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Final cache state
    print(f"\n📊 Final Cache State:")
    final_models = check_model_cache()
    
    if len(final_models) > len(initial_models or []):
        print(f"✨ Successfully downloaded {len(final_models) - len(initial_models or [])} new models!")

def show_model_info():
    """Show information about available models"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/")
        if response.status_code == 200:
            models = response.json()
            print("\n🔧 Available Models from API:")
            for model in models:
                print(f"   • {model['name']}: {model['description']}")
                print(f"     Loaded: {model['loaded']}, Device: {model['device']}")
        else:
            print(f"❌ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting models: {str(e)}")

if __name__ == "__main__":
    print("Model Downloading and Caching Test")
    print("This script tests the automatic downloading and caching of language-specific models")
    print("\nMake sure your Django server is running on localhost:8000")
    
    input("\nPress Enter to continue...")
    
    # Show available models
    show_model_info()
    
    # Run the download tests
    test_model_downloading()
    
    print("\n" + "=" * 60)
    print("Test completed! Check the 'models' directory to see cached models.") 