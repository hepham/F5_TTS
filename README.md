# Django F5-TTS API

A Django REST API for F5-TTS (Flow Matching Text-to-Speech) and E2-TTS (Embarrassingly Easy Text-to-Speech) models, providing high-quality text-to-speech generation capabilities through RESTful endpoints.

## Features

- 🎯 **RESTful API** - Clean REST endpoints for text-to-speech generation
- 🎤 **Multiple Models** - Support for F5-TTS, E2-TTS, and custom models
- 📁 **File Upload** - Upload reference audio files for voice cloning
- ⚡ **High Performance** - Optimized for batch processing and production use
- 📚 **API Documentation** - Automatic Swagger/OpenAPI documentation
- 🔧 **Configurable** - Adjustable parameters for speed, silence removal, etc.
- 🎵 **Audio Output** - Returns base64-encoded WAV audio files

## Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 4GB RAM (8GB+ recommended)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd F5-TTS/Django
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   python start_server.py
   ```

   Or manually:
   ```bash
   python manage.py migrate
   python manage.py runserver 0.0.0.0:8000
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8000/api/docs/
   - Health Check: http://localhost:8000/api/tts/health/
   - Models List: http://localhost:8000/api/tts/models/

## API Endpoints

### 🏥 Health Check
```
GET /api/tts/health/
```

Check if the API is running and get available models.

**Response:**
```json
{
  "status": "healthy",
  "message": "F5-TTS API is running",
  "available_models": ["F5-TTS", "E2-TTS", "Custom"]
}
```

### 📋 Available Models
```
GET /api/tts/models/
```

Get the list of available TTS models.

**Response:**
```json
{
  "models": ["F5-TTS", "E2-TTS", "Custom"]
}
```

### 🎤 Text-to-Speech Generation
```
POST /api/tts/generate/
```

Generate speech from text using reference audio for voice cloning.

**Request (multipart/form-data):**
- `ref_audio` (file, required): Reference audio file (WAV, MP3, etc.)
- `text` (string, required): Text to convert to speech
- `model` (string, optional): Model to use ("F5-TTS", "E2-TTS", "Custom"). Default: "F5-TTS"
- `ref_text` (string, optional): Reference text (auto-transcribed if not provided)
- `remove_silence` (boolean, optional): Remove silences from output. Default: false
- `cross_fade_duration` (float, optional): Cross-fade duration in seconds. Default: 0.15
- `speed` (float, optional): Speech speed multiplier. Default: 1.0
- `custom_model_path` (string, optional): Path to custom model (required if model="Custom")
- `vocab_path` (string, optional): Path to vocabulary file (for custom models)

**Response:**
```json
{
  "audio_data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "sample_rate": 24000,
  "ref_text": "This is the processed reference text",
  "message": "Speech generation successful",
  "model_used": "F5-TTS"
}
```

The `audio_data` field contains base64-encoded WAV audio data.

## Usage Examples

### Python Example

```python
import requests
import base64

# Prepare the request
url = "http://localhost:8000/api/tts/generate/"
files = {'ref_audio': open('reference_audio.wav', 'rb')}
data = {
    'text': 'Hello, this is a test of the F5-TTS API!',
    'model': 'F5-TTS',
    'remove_silence': False,
    'speed': 1.0
}

# Make the request
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    
    # Decode and save the audio
    audio_data = base64.b64decode(result['audio_data'])
    with open('generated_audio.wav', 'wb') as f:
        f.write(audio_data)
    
    print(f"Generated audio saved! Sample rate: {result['sample_rate']} Hz")
else:
    print(f"Error: {response.text}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/tts/generate/" \
     -F "ref_audio=@reference_audio.wav" \
     -F "text=Hello, this is a test of the F5-TTS API!" \
     -F "model=F5-TTS" \
     -F "speed=1.0"
```

### JavaScript/Fetch Example

```javascript
const formData = new FormData();
formData.append('ref_audio', audioFile); // File object
formData.append('text', 'Hello, this is a test!');
formData.append('model', 'F5-TTS');

fetch('http://localhost:8000/api/tts/generate/', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    // Decode base64 audio data
    const audioData = atob(data.audio_data);
    const audioBlob = new Blob([audioData], { type: 'audio/wav' });
    
    // Create and play audio
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
});
```

## Testing

### Test Script

Use the included test script to verify the API:

```bash
# Basic test
python test_django_api.py

# Test with custom audio and text
python test_django_api.py --audio my_audio.wav --text "Custom test message"

# Benchmark with 10 iterations
python test_django_api.py --benchmark 10

# Test with different host/port
python test_django_api.py --host 192.168.1.100 --port 8080
```

### Test Script Options

- `--host`: API host (default: localhost)
- `--port`: API port (default: 8000)
- `--audio`: Reference audio file (creates test audio if not provided)
- `--text`: Text to generate (default: "Hello, this is a test of the F5-TTS API!")
- `--model`: Model to use (F5-TTS, E2-TTS)
- `--benchmark N`: Run benchmark with N iterations

## Server Configuration

### Start Server Options

```bash
python start_server.py --help
```

Options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--debug`: Enable debug mode
- `--no-migrate`: Skip database migrations
- `--no-static`: Skip static file collection
- `--check-deps`: Only check dependencies and exit

### Environment Variables

- `DJANGO_SETTINGS_MODULE`: Django settings module (default: tts_api.settings)
- `DJANGO_LOG_LEVEL`: Django logging level (default: INFO)

## Model Support

### F5-TTS (Default)
- High-quality flow matching TTS
- Good for most use cases
- Fast inference

### E2-TTS
- Alternative TTS model
- Different voice characteristics
- May require additional setup

### Custom Models
- Load your own trained models
- Requires model checkpoint and vocabulary files
- Support for custom configurations

## Performance Optimization

### GPU Support
The API automatically uses CUDA if available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Memory Management
- Models are loaded once and cached
- Automatic cleanup of temporary files
- Efficient audio processing

### Batch Processing
- Automatic text chunking for long inputs
- Cross-fade between audio segments
- Optimized for production workloads

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK`: Successful generation
- `400 Bad Request`: Invalid input parameters
- `500 Internal Server Error`: Server-side errors

Error response format:
```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 400
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/api/docs/
- ReDoc: http://localhost:8000/api/redoc/
- JSON Schema: http://localhost:8000/api/schema/

## Deployment

### Development
```bash
python start_server.py --debug
```

### Production
Consider using:
- **Gunicorn**: `gunicorn tts_api.wsgi:application --bind 0.0.0.0:8000`
- **uWSGI**: `uwsgi --http :8000 --module tts_api.wsgi`
- **Docker**: See Docker configuration below

### Docker Deployment (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t django-f5tts .
docker run -p 8000:8000 django-f5tts
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed and F5-TTS is in the Python path
2. **CUDA Issues**: Verify PyTorch CUDA installation
3. **Audio Format Issues**: Ensure reference audio is in a supported format (WAV, MP3, etc.)
4. **Memory Issues**: Reduce batch size or use CPU-only mode for lower memory usage

### Logs

Check Django logs for detailed error information:
```bash
python manage.py runserver --verbosity=2
```

### Debug Mode

Enable debug mode for detailed error messages:
```bash
python start_server.py --debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project follows the same license as F5-TTS. Please refer to the main project license for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check existing issues in the repository
4. Create a new issue with detailed information

## Changelog

### v1.0.0
- Initial Django F5-TTS API implementation
- Support for F5-TTS and E2-TTS models
- RESTful API with comprehensive documentation
- Test script and benchmarking tools
- Production-ready configuration "# F5_TTS" 
