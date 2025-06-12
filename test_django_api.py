#!/usr/bin/env python3
"""
Test script for Django F5-TTS API

This script demonstrates how to use the Django F5-TTS REST API for text-to-speech generation.
"""

import requests
import base64
import io
import argparse
import time
from pathlib import Path


def test_health_check(base_url):
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{base_url}/health/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_models_endpoint(base_url):
    """Test the models endpoint"""
    print("\n=== Testing Models Endpoint ===")
    try:
        response = requests.get(f"{base_url}/models/")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Available models: {data.get('models', [])}")
        return response.status_code == 200
    except Exception as e:
        print(f"Models endpoint failed: {e}")
        return False


def test_tts_generation(base_url, audio_file, text, model_name="F5-TTS", language=None):
    """Test TTS generation endpoint"""
    print(f"\n=== Testing TTS Generation (Model: {model_name}) ===")
    if language:
        print(f"Language: {language}")
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        return False, None
    
    start_time = time.time()
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'ref_audio': (Path(audio_file).name, f, 'audio/wav')}
            data = {
                'ref_text': '',  # Will be auto-transcribed
                'text': text,  # Fixed: API expects 'text' not 'gen_text'
                'model': model_name,
                'remove_silence': True,
                'speed': 1.0
            }
            
            # Add language parameter if specified
            if language:
                data['language'] = language
            
            response = requests.post(f"{base_url}/generate/", files=files, data=data)
        
        generation_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Generation Time: {generation_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            # Get audio data
            audio_b64 = result.get('audio_data')
            if audio_b64:
                # Decode and save audio
                audio_data = base64.b64decode(audio_b64)
                output_file = f"output_{int(time.time())}.wav"
                
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                print(f"✅ Audio saved as: {output_file}")
                print(f"Sample Rate: {result.get('sample_rate', 'N/A')}")
                print(f"Duration: {result.get('duration', 'N/A')}s")
                
                return True, output_file
            else:
                print("❌ No audio data in response")
                return False, None
        else:
            print(f"❌ Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return False, None


def benchmark_api(base_url, iterations=10, audio_file=None, text=None, model_name="F5-TTS", language=None):
    """Benchmark the TTS API"""
    print(f"\n=== Benchmarking API ({iterations} iterations) ===")
    print(f"Model: {model_name}")
    if language:
        print(f"Language: {language}")
    print(f"Text: {text}")
    
    if not audio_file or not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        print("Please provide a valid reference audio file using --audio parameter")
        return
    
    times = []
    success_count = 0
    
    # Create output directory for benchmark results
    output_dir = Path("benchmark_outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}", end=" - ")
        
        try:
            start_time = time.time()
            success, output_file = test_tts_generation(base_url, audio_file, text, model_name, language)
            generation_time = time.time() - start_time
            
            if success and output_file:
                # Move the generated file to benchmark output directory
                benchmark_filename = f"benchmark_iter_{i+1:03d}_{timestamp}.wav"
                benchmark_path = output_dir / benchmark_filename
                
                # Move file to benchmark directory
                if Path(output_file).exists():
                    Path(output_file).rename(benchmark_path)
                    print(f"✅ {generation_time:.2f}s -> {benchmark_filename}")
                else:
                    print(f"❌ File not found: {output_file}")
                    continue
                
                times.append(generation_time)
                success_count += 1
            else:
                print(f"❌ Failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Calculate statistics
    if times:
        avg_time = sum(times) / len(times)
        median_time = sorted(times)[len(times) // 2]
        min_time = min(times)
        max_time = max(times)
        
        # Calculate standard deviation
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5
        
        # Calculate throughput (characters per second)
        if text:
            text_length = len(text)
            avg_chars_per_sec = text_length / avg_time if avg_time > 0 else 0
        else:
            avg_chars_per_sec = 0
        
        print(f"\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(f"Total Iterations: {iterations}")
        print(f"Successful: {success_count}")
        print(f"Failed: {iterations - success_count}")
        print(f"Success Rate: {(success_count/iterations)*100:.1f}%")
        print(f"\nTiming Statistics:")
        print(f"  Average: {avg_time:.2f}s")
        print(f"  Median:  {median_time:.2f}s")
        print(f"  Min:     {min_time:.2f}s")
        print(f"  Max:     {max_time:.2f}s")
        print(f"  Std Dev: {std_dev:.2f}s")
        if avg_chars_per_sec > 0:
            print(f"  Throughput: {avg_chars_per_sec:.1f} chars/sec")
        
        print(f"\n📁 All audio files saved in: {output_dir}")
        
        # Create summary file
        summary_file = output_dir / f"benchmark_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Django F5-TTS API Benchmark Summary\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Text: {text}\n\n")
            f.write(f"Results:\n")
            f.write(f"Total Iterations: {iterations}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {iterations - success_count}\n")
            f.write(f"Success Rate: {(success_count/iterations)*100:.1f}%\n\n")
            f.write(f"Timing Statistics:\n")
            f.write(f"Average: {avg_time:.2f}s\n")
            f.write(f"Median: {median_time:.2f}s\n")
            f.write(f"Min: {min_time:.2f}s\n")
            f.write(f"Max: {max_time:.2f}s\n")
            f.write(f"Std Dev: {std_dev:.2f}s\n")
            if avg_chars_per_sec > 0:
                f.write(f"Throughput: {avg_chars_per_sec:.1f} chars/sec\n")
            f.write(f"\nGenerated Files:\n")
            for i in range(success_count):
                f.write(f"benchmark_iter_{i+1:03d}_{timestamp}.wav\n")
        
        print(f"📄 Summary saved as: {summary_file}")
        
    else:
        print("\n❌ No successful generations in benchmark!")
        print("Please check your server and model configuration.")


def test_gpu_status(base_url):
    """Test the GPU status endpoint"""
    print("\n=== Testing GPU Status ===")
    try:
        response = requests.get(f"{base_url}/gpu-status/")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Device: {data.get('device', 'N/A')}")
        print(f"GPU Available: {data.get('gpu_available', False)}")
        
        if data.get('gpu_available'):
            print(f"GPU Name: {data.get('gpu_name', 'N/A')}")
            print(f"Memory Allocated: {data.get('memory_allocated', 'N/A')}")
            print(f"Memory Reserved: {data.get('memory_reserved', 'N/A')}")
            print(f"Memory Total: {data.get('memory_total', 'N/A')}")
        else:
            print(f"Message: {data.get('message', 'N/A')}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"GPU status check failed: {e}")
        return False


def clear_gpu_cache(base_url):
    """Clear GPU cache"""
    print("\n=== Clearing GPU Cache ===")
    try:
        response = requests.post(f"{base_url}/gpu-status/")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Message: {data.get('message', 'No message')}")
        return response.status_code == 200
    except Exception as e:
        print(f"GPU cache clear failed: {e}")
        return False


def test_multilingual_tts(base_url, audio_file):
    """Test TTS generation for each supported language"""
    print(f"\n=== Testing Multilingual TTS Generation ===")
    
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found!")
        return False
    
    # Test cases for each supported language
    language_tests = [
        {
            "language": "en",
            "text": "Hello, this is a test of the English language model.",
            "expected_model": "SWivid/F5-TTS_v1",
            "description": "English"
        },
        {
            "language": "zh",
            "text": "你好，这是中文语言模型的测试。",
            "expected_model": "SWivid/F5-TTS_v1", 
            "description": "Chinese"
        },
        {
            "language": "es",
            "text": "Hola, esta es una prueba del modelo de idioma español.",
            "expected_model": "jpgallegoar/F5-Spanish",
            "description": "Spanish"
        },
        {
            "language": "fr",
            "text": "Bonjour, ceci est un test du modèle de langue française.",
            "expected_model": "RASPIAUDIO/F5-French-MixedSpeakers-reduced",
            "description": "French"
        },
        {
            "language": "it",
            "text": "Ciao, questo è un test del modello di lingua italiana.",
            "expected_model": "alien79/F5-TTS-italian",
            "description": "Italian"
        },
        {
            "language": "ja",
            "text": "こんにちは、これは日本語言語モデルのテストです。",
            "expected_model": "Jmica/F5TTS/JA_21999120",
            "description": "Japanese"
        },
        {
            "language": "ru",
            "text": "Привет, это тест модели русского языка.",
            "expected_model": "hotstone228/F5-TTS-Russian",
            "description": "Russian"
        },
        {
            "language": "hi",
            "text": "नमस्ते, यह हिंदी भाषा मॉडल का परीक्षण है।",
            "expected_model": "SPRINGLab/F5-Hindi-24KHz",
            "description": "Hindi"
        },
        {
            "language": "fi",
            "text": "Hei, tämä on suomen kielen mallin testi.",
            "expected_model": "AsmoKoskinen/F5-TTS_Finnish_Model",
            "description": "Finnish"
        }
    ]
    
    print(f"Testing {len(language_tests)} different languages...")
    
    # Create output directory for multilingual results
    output_dir = Path("multilingual_outputs")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    results = []
    
    for i, test_case in enumerate(language_tests, 1):
        print(f"\n🧪 Test {i}/{len(language_tests)}: {test_case['description']}")
        print(f"   Language: {test_case['language']}")
        print(f"   Expected Model: {test_case['expected_model']}")
        print(f"   Text: {test_case['text']}")
        
        start_time = time.time()
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'ref_audio': (Path(audio_file).name, f, 'audio/wav')}
                data = {
                    'ref_text': '',  # Will be auto-transcribed
                    'text': test_case['text'],
                    'language': test_case['language'],  # Specify language explicitly
                    'model': 'Auto',  # Let the system choose based on language
                    'remove_silence': True,
                    'speed': 1.0
                }
                
                response = requests.post(f"{base_url}/generate/", files=files, data=data)
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Get audio data
                audio_b64 = result.get('audio_data')
                if audio_b64:
                    # Decode and save audio
                    audio_data = base64.b64decode(audio_b64)
                    output_filename = f"multilingual_{test_case['language']}_{timestamp}.wav"
                    output_path = output_dir / output_filename
                    
                    with open(output_path, 'wb') as f:
                        f.write(audio_data)
                    
                    print(f"   ✅ Success! (took {generation_time:.1f}s)")
                    print(f"   💾 Audio saved: {output_path}")
                    print(f"   📊 Sample Rate: {result.get('sample_rate', 'N/A')}")
                    print(f"   ⏱️  Duration: {result.get('duration', 'N/A')}s")
                    
                    # Check if correct model was used (from response metadata if available)
                    processing_info = result.get('processing_info', {})
                    if processing_info:
                        detected_lang = processing_info.get('detected_language', 'N/A')
                        print(f"   🔍 Detected Language: {detected_lang}")
                    
                    results.append({
                        'language': test_case['language'],
                        'description': test_case['description'],
                        'success': True,
                        'time': generation_time,
                        'file': output_filename,
                        'expected_model': test_case['expected_model']
                    })
                else:
                    print("   ❌ No audio data in response")
                    results.append({
                        'language': test_case['language'],
                        'description': test_case['description'],
                        'success': False,
                        'error': 'No audio data'
                    })
            else:
                print(f"   ❌ Failed: {response.status_code}")
                print(f"   📄 Error: {response.text}")
                results.append({
                    'language': test_case['language'],
                    'description': test_case['description'],
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                })
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'language': test_case['language'],
                'description': test_case['description'],
                'success': False,
                'error': str(e)
            })
    
    # Generate summary
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\n" + "="*60)
    print("MULTILINGUAL TEST SUMMARY")
    print("="*60)
    print(f"Total Languages Tested: {len(language_tests)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {(len(successful)/len(language_tests))*100:.1f}%")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average Generation Time: {avg_time:.2f}s")
        
        print(f"\n✅ Successful Languages:")
        for result in successful:
            print(f"   • {result['description']} ({result['language']}): {result['time']:.1f}s -> {result['file']}")
    
    if failed:
        print(f"\n❌ Failed Languages:")
        for result in failed:
            print(f"   • {result['description']} ({result['language']}): {result['error']}")
    
    # Save summary file
    summary_file = output_dir / f"multilingual_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Django F5-TTS Multilingual Test Summary\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Languages: {len(language_tests)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Success Rate: {(len(successful)/len(language_tests))*100:.1f}%\n\n")
        
        if successful:
            f.write(f"Successful Tests:\n")
            for result in successful:
                f.write(f"  {result['description']} ({result['language']}): {result['time']:.1f}s\n")
                f.write(f"    Expected Model: {result['expected_model']}\n")
                f.write(f"    Output File: {result['file']}\n\n")
        
        if failed:
            f.write(f"Failed Tests:\n")
            for result in failed:
                f.write(f"  {result['description']} ({result['language']}): {result['error']}\n")
    
    print(f"\n📁 All audio files saved in: {output_dir}")
    print(f"📄 Summary saved as: {summary_file}")
    
    return len(successful) > 0


def test_debug_model_detection(base_url, model_name="Jmica/F5TTS/JA_21999120"):
    """Test debug model detection endpoint"""
    print(f"\n=== Testing Debug Model Detection ===")
    print(f"Model: {model_name}")
    
    try:
        params = {'model': model_name}
        response = requests.get(f"{base_url}/debug/model/", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Debug successful!")
            print(f"Model exists locally: {data.get('exists_locally', 'Unknown')}")
            print(f"Message: {data.get('message', 'No message')}")
            print("📋 Check console/logs for detailed debug information")
            return True
        else:
            print(f"❌ Debug failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Debug error: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Django F5-TTS API')
    parser.add_argument('--url', default='http://localhost:8000/api/tts', help='API base URL')
    parser.add_argument('--audio', help='Path to reference audio file (.wav)')
    parser.add_argument('--text', default='I remember the first time I walked through the old streets of the city. Everything felt so different from what I was used to!', 
                       help='Text to convert to speech')
    parser.add_argument('--model', default='F5-TTS', choices=['F5-TTS', 'E2-TTS'], help='TTS model to use')
    parser.add_argument('--benchmark', type=int, help='Run benchmark with specified number of iterations')
    parser.add_argument('--multilingual', action='store_true', help='Test all supported languages with one sentence each')
    parser.add_argument('--gpu-status', action='store_true', help='Check GPU status only')
    parser.add_argument('--clear-cache', action='store_true', help='Clear GPU cache only')
    parser.add_argument('--debug-model', action='store_true', help='Debug local model detection')
    parser.add_argument('--language', choices=['en', 'zh', 'es', 'fr', 'it', 'ja', 'ru', 'hi', 'fi'], 
                       help='Specify language for TTS generation (en=English, zh=Chinese, es=Spanish, fr=French, it=Italian, ja=Japanese, ru=Russian, hi=Hindi, fi=Finnish)')
    
    args = parser.parse_args()
    
    # Ensure base_url doesn't end with slash to avoid double slashes
    base_url = args.url.rstrip('/')
    
    print(f"Testing Django F5-TTS API at: {base_url}")
    
    # Handle special modes first
    if args.gpu_status:
        print("Checking GPU status...")
        test_gpu_status(base_url)
        return
        
    if args.clear_cache:
        print("Clearing GPU cache...")
        clear_gpu_cache(base_url)
        return
    
    if args.debug_model:
        test_debug_model_detection(base_url)
        return
    
    # For other operations, audio file is required
    if not args.audio:
        print("Error: --audio parameter is required")
        print("Please provide a reference audio file: --audio your_voice.wav")
        return
        
    if not Path(args.audio).exists():
        print(f"Error: Audio file '{args.audio}' not found!")
        return
    
    print(f"Reference audio file: {args.audio}")
    
    print("\n" + "="*50)
    print("DJANGO F5-TTS API TESTING")
    print("="*50)
    
    # Run different test modes
    if args.benchmark:
        benchmark_api(base_url, args.benchmark, args.audio, args.text, args.model, args.language)
    elif args.multilingual:
        # Test all supported languages
        print("🌍 Running multilingual tests for all supported languages...")
        test_multilingual_tts(base_url, args.audio)
    else:
        # Basic functionality tests
        health_ok = test_health_check(base_url)
        models_ok = test_models_endpoint(base_url)
        gpu_ok = test_gpu_status(base_url)
        
        if not (health_ok and models_ok):
            print("\n❌ API basic tests failed. Please check if the server is running.")
            return
        
        # Single TTS test
        audio_ok, output_file = test_tts_generation(base_url, args.audio, args.text, args.model, args.language)
        
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Health Check: {'✅' if health_ok else '❌'}")
        print(f"Models Endpoint: {'✅' if models_ok else '❌'}")
        print(f"GPU Status: {'✅' if gpu_ok else '❌'}")
        print(f"TTS Generation: {'✅' if audio_ok else '❌'}")
        
        if audio_ok and output_file:
            print(f"\n🎵 Generated audio saved as: {output_file}")
            print("You can now listen to the generated speech!")
            print("\n💡 Tip: Use --multilingual to test all supported languages!")


if __name__ == '__main__':
    main() 