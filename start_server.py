#!/usr/bin/env python3
"""
Django F5-TTS Server Startup Script

This script helps start the Django F5-TTS server with proper configuration.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def setup_environment():
    """Set up the environment for the Django server"""
    # Add the parent directory to Python path for F5-TTS imports
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tts_api.settings')


def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch with CUDA 12.4 support...")
    try:
        cmd = [
            sys.executable, '-m', 'pip', 'install', 
            'torch==2.4.0+cu124', 
            'torchaudio==2.4.0+cu124',
            '--extra-index-url', 'https://download.pytorch.org/whl/cu124'
        ]
        subprocess.run(cmd, check=True)
        print("PyTorch with CUDA installed successfully!")
        
        # Test CUDA availability
        test_cuda()
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyTorch: {e}")
        return False
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("CUDA is not available. F5-TTS will run on CPU (slower).")
    except ImportError:
        print("PyTorch not installed yet.")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    missing_deps = []
    
    # Critical dependencies
    critical_deps = {
        'django': 'Django',
        'rest_framework': 'djangorestframework', 
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'soundfile': 'soundfile',
        'transformers': 'transformers'
    }
    
    for module, package in critical_deps.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Run with --install-deps to install missing packages")
        return False
    
    print("Dependency check completed successfully!")
    return True


def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies...")
    try:
        # Install regular requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("Regular dependencies installed!")
        
        # Install PyTorch with CUDA
        if not install_pytorch():
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False
    
    return True


def migrate_database():
    """Run Django migrations"""
    print("Running database migrations...")
    try:
        subprocess.run([sys.executable, 'manage.py', 'migrate'], check=True)
        print("Migrations completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Migration failed: {e}")
        return False
    return True


def start_server(host='localhost', port=8000, debug=False):
    """Start the Django development server"""
    print(f"Starting Django server at http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        if debug:
            # Use Django's built-in development server
            subprocess.run([sys.executable, 'manage.py', 'runserver', f'{host}:{port}'])
        else:
            # Use uvicorn for better performance
            subprocess.run([
                sys.executable, '-m', 'uvicorn', 
                'tts_api.asgi:application',
                '--host', host,
                '--port', str(port),
                '--reload'
            ])
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start server: {e}")


def main():
    parser = argparse.ArgumentParser(description='Django F5-TTS Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--install-pytorch', action='store_true', help='Install PyTorch with CUDA')
    parser.add_argument('--test-cuda', action='store_true', help='Test CUDA availability')
    parser.add_argument('--migrate', action='store_true', help='Run database migrations only')
    parser.add_argument('--debug', action='store_true', help='Use Django dev server instead of uvicorn')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Handle different modes
    if args.check_deps:
        check_dependencies()
        return
    
    if args.install_deps:
        if install_dependencies():
            print("All dependencies installed successfully!")
        return
    
    if args.install_pytorch:
        install_pytorch()
        return
        
    if args.test_cuda:
        test_cuda()
        return
    
    if args.migrate:
        migrate_database()
        return
    
    # Check dependencies before starting server
    if not check_dependencies():
        print("Some dependencies are missing. Install them first.")
        return
    
    # Run migrations
    if not migrate_database():
        return
    
    # Start server
    start_server(args.host, args.port, args.debug)


if __name__ == '__main__':
    main() 