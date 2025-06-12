# Technical Context: F5-TTS Django REST API

## Technology Stack

### Core Framework
- **Django 4.2.7**: Web framework for API backend
- **Django REST Framework 3.14.0**: RESTful API functionality
- **Python 3.x**: Primary programming language

### API & Documentation
- **drf-yasg 1.21.7**: Swagger/OpenAPI documentation
- **django-cors-headers 4.3.1**: Cross-origin resource sharing

### Audio Processing
- **torch >= 2.0.0**: PyTorch for deep learning models
- **torchaudio >= 2.0.0**: Audio processing for PyTorch
- **numpy >= 1.21.0**: Numerical computations
- **soundfile >= 0.12.1**: Audio file I/O

### TTS Models
- **f5-tts**: Core F5-TTS model package
- **transformers >= 4.30.0**: Hugging Face transformers
- **cached-path >= 1.5.0**: Model caching utilities

**Available Language-Specific Models (Discovered):**
- **English & Chinese**: SWivid/F5-TTS_v1 (official base model)
- **Finnish**: AsmoKoskinen/F5-TTS_Finnish_Model  
- **French**: RASPIAUDIO/F5-French-MixedSpeakers-reduced
- **Hindi**: SPRINGLab/F5-Hindi-24KHz
- **Italian**: alien79/F5-TTS-italian
- **Japanese**: Jmica/F5TTS/JA_21999120
- **Russian**: hotstone228/F5-TTS-Russian
- **Spanish**: jpgallegoar/F5-Spanish

**Integration Strategy:**
- Use Hugging Face Hub for model downloads
- Implement language parameter in API
- Cache downloaded models locally
- Support automatic model discovery

### Utilities
- **click >= 8.0.0**: Command-line interface
- **uvicorn >= 0.20.0**: ASGI server (optional)

## Development Environment

### Platform
- **Operating System**: Windows 10/11 (development)
- **Python Version**: 3.x (preferably 3.8-3.12)
- **Shell**: PowerShell (Windows)

### Dependencies Installation
```bash
pip install -r requirements.txt
```

### Current Dependency Issues
1. **F5-TTS Package**: Installation fails on Windows due to C++ compiler requirements
2. **NumPy Compilation**: Requires Microsoft Visual C++ Build Tools
3. **Python 3.13**: Compatibility issues with distutils module

### Alternative Solutions (New Discovery)
1. **Hugging Face Models**: Direct download of language-specific models
2. **Transformers Integration**: Use transformers library instead of f5-tts package
3. **Pre-trained Models**: Access to 8 language-specific models available
4. **Model Hub Integration**: Leverage Hugging Face Hub for model management

### Workarounds Implemented
- **Dummy Audio Generation**: Sine wave generation for testing
- **Placeholder Models**: Mock model loading for API structure
- **Service Layer Abstraction**: Ready for real model integration
- **Language Model Discovery**: Comprehensive list of available models

## Configuration

### Django Settings Structure
```python
# Core Django
DEBUG = True  # Development mode
ALLOWED_HOSTS = ['*']  # Open for development
SECRET_KEY = '...'  # Should be environment variable in production

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
}

# File Uploads
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB

# CORS
CORS_ALLOW_ALL_ORIGINS = True  # Development only
```

### Environment Variables (Planned)
```bash
# Django Configuration
DJANGO_DEBUG=True
DJANGO_SECRET_KEY=your-secret-key
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# Model Configuration
F5TTS_MODEL_PATH=/path/to/models
ENABLE_F5TTS=True
ENABLE_E2TTS=True

# Performance
MAX_AUDIO_DURATION=300  # seconds
CONCURRENT_REQUESTS=5
```

## Project Structure

### Django Project Layout
```
tts_api/                    # Django project root
├── manage.py              # Django management script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── tts_api/              # Django project settings
│   ├── __init__.py
│   ├── settings.py       # Main configuration
│   ├── urls.py          # URL routing
│   ├── wsgi.py          # WSGI application
│   └── asgi.py          # ASGI application
└── tts/                  # TTS application
    ├── __init__.py
    ├── models.py         # Django models (minimal)
    ├── views.py          # API views
    ├── urls.py           # App URL routing
    ├── serializers.py    # Request/response serializers
    ├── services.py       # TTS business logic
    ├── apps.py           # App configuration
    ├── admin.py          # Django admin (unused)
    ├── tests.py          # Unit tests
    └── migrations/       # Database migrations
```

### Memory Bank Structure
```
memory-bank/
├── projectbrief.md       # Project overview and requirements
├── productContext.md     # Product goals and user experience
├── systemPatterns.md     # Architecture and design patterns
├── techContext.md        # This file - technical details
├── activeContext.md      # Current work and decisions
└── progress.md          # Implementation status
```

## Development Setup

### Quick Start
1. **Clone Repository**: Get project code
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run Migrations**: `python manage.py migrate`
4. **Start Server**: `python manage.py runserver`
5. **Access API**: `http://localhost:8000`

### Development Workflow
1. **Code Changes**: Edit Python files
2. **Test Locally**: Use Django development server
3. **API Testing**: Use Swagger UI at `http://localhost:8000/`
4. **Manual Testing**: Use curl or Python requests

## API Endpoints

### Core Endpoints
- **POST** `/api/tts/generate/`: TTS generation with file upload
- **GET** `/api/tts/health/`: Health check and status
- **GET** `/api/tts/models/`: Available model listing

### Documentation Endpoints
- **GET** `/`: Swagger UI (development)
- **GET** `/api/docs/`: Swagger UI
- **GET** `/api/redoc/`: ReDoc documentation
- **GET** `/api/schema/`: OpenAPI schema JSON

## Performance Considerations

### Memory Management
- **Model Caching**: Single instance per model type
- **File Cleanup**: Automatic temporary file removal
- **Memory Limits**: 50MB file upload limit

### Response Times
- **Model Loading**: One-time cost on first request
- **Audio Generation**: ~5-30 seconds depending on text length
- **File Upload**: Limited by network and file size

### Scalability Constraints
- **Single Process**: Django development server
- **Memory Usage**: Large models consume significant RAM
- **Concurrent Requests**: Limited by model thread safety

## Security Considerations

### File Upload Security
- **File Size Limits**: 50MB maximum
- **Temporary Storage**: Auto-cleanup prevents disk filling
- **File Type Validation**: Audio format checking
- **Path Sanitization**: Prevent directory traversal

### API Security
- **Input Validation**: All parameters validated
- **Error Messages**: No sensitive information leaked
- **CORS Policy**: Open for development (restrict in production)
- **Rate Limiting**: Not implemented (future requirement)

## Deployment Considerations

### Production Requirements
- **ASGI Server**: Uvicorn or Gunicorn with workers
- **Reverse Proxy**: Nginx for static files and SSL
- **Environment Variables**: Secure configuration
- **Model Storage**: Persistent volume for model files

### Infrastructure Needs
- **Memory**: 8GB+ RAM for model loading
- **Storage**: Space for model checkpoints (~1-2GB)
- **CPU**: Multi-core recommended for audio processing
- **Network**: Sufficient bandwidth for audio uploads

## Testing Strategy

### Current Testing
- **Manual Testing**: Swagger UI for endpoint testing
- **Dummy Audio**: Sine wave generation for validation
- **Error Scenarios**: Parameter validation testing

### Future Testing
- **Unit Tests**: Service layer business logic
- **Integration Tests**: Full API endpoint testing
- **Performance Tests**: Load testing with real models
- **Audio Quality Tests**: Output validation with real TTS

## Known Issues & Limitations

### Current Issues
1. **F5-TTS Installation**: Windows compilation errors
2. **Model Loading**: Using placeholder implementations
3. **Audio Quality**: Dummy sine wave output only

### Technical Debt
1. **Error Handling**: Some edge cases not covered
2. **Logging**: Could be more detailed
3. **Configuration**: Hardcoded values should be configurable
4. **Testing**: No automated tests yet

### Future Improvements
1. **Real Model Integration**: Resolve F5-TTS installation
2. **Performance Optimization**: Model loading and caching
3. **Security Hardening**: Rate limiting, authentication
4. **Monitoring**: Health checks and metrics 