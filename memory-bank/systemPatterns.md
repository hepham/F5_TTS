# System Patterns: F5-TTS Django REST API

## Architecture Overview

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Django REST    │───▶│   TTS Service   │
│                 │    │      API        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  File Storage   │    │  TTS Models     │
                       │  (Temporary)    │    │  (F5/E2/Custom) │
                       └─────────────────┘    └─────────────────┘
```

### Layer Architecture

#### API Layer (Django REST Framework)
- **Views**: Handle HTTP requests/responses
- **Serializers**: Data validation and transformation
- **URL Routing**: API endpoint configuration
- **Middleware**: CORS, authentication (future)

#### Service Layer 
- **TTS Service**: Core business logic for speech generation
- **Model Management**: Loading and caching of TTS models
- **Audio Processing**: File handling and audio format conversion

#### Data Layer
- **Temporary File Storage**: Uploaded reference audio files
- **Model Storage**: TTS model checkpoints and vocabularies
- **Response Caching**: Base64 audio data (in-memory)

## Key Design Patterns

### Service Pattern
```python
# Single responsibility for TTS operations
class TTSService:
    def generate_speech_base64() -> Tuple[str, int, str]
    def get_available_models() -> list
    def _load_model() -> Any
```

**Benefits:**
- Separation of concerns between API and business logic
- Testable business logic independent of Django
- Reusable service across different endpoints

### Factory Pattern (Models)
```python
def _get_model(model_name: str) -> Any:
    if model_name == "F5-TTS":
        return self._load_f5tts()
    elif model_name == "E2-TTS":
        return self._load_e2tts()
    # ...
```

**Benefits:**
- Consistent model loading interface
- Easy to add new model types
- Centralized model configuration

### Singleton Pattern (Service Instance)
```python
_tts_service_instance = None

def get_tts_service() -> TTSService:
    global _tts_service_instance
    if _tts_service_instance is None:
        _tts_service_instance = TTSService()
    return _tts_service_instance
```

**Benefits:**
- Single model instance across requests
- Memory efficiency for large models
- Consistent state management

### Strategy Pattern (Audio Processing)
```python
# Different processing strategies for different models
def generate_speech(model_name: str, ...):
    model = self._get_model(model_name)
    # Strategy varies by model type
```

## Component Relationships

### Core Components

#### TTSGenerationView
- **Responsibility**: Handle TTS generation API requests
- **Dependencies**: TTSService, TTSRequestSerializer, TTSResponseSerializer
- **Input**: Multipart form data (audio file + parameters)
- **Output**: JSON response with base64 audio

#### TTSService
- **Responsibility**: Core TTS business logic
- **Dependencies**: F5-TTS models, numpy, soundfile
- **State**: Cached model instances
- **Methods**: Speech generation, model management

#### Serializers
- **TTSRequestSerializer**: Input validation and transformation
- **TTSResponseSerializer**: Output formatting
- **Error handling**: Consistent error response structure

### Data Flow Patterns

#### Request Processing Flow
1. **File Upload**: Client uploads reference audio
2. **Validation**: Serializer validates all input parameters
3. **Temporary Storage**: Audio saved to temporary file
4. **Model Loading**: Appropriate TTS model loaded/retrieved
5. **Speech Generation**: TTS service processes audio and text
6. **Encoding**: Audio converted to base64 format
7. **Response**: JSON response with audio data
8. **Cleanup**: Temporary files removed

#### Error Handling Flow
1. **Input Validation**: Serializer catches invalid data
2. **Business Logic Errors**: Service catches processing errors
3. **System Errors**: View catches unexpected exceptions
4. **Consistent Responses**: All errors follow same JSON structure

## Configuration Patterns

### Environment-Based Configuration
```python
# settings.py patterns
DEBUG = os.getenv('DJANGO_DEBUG', 'False').lower() == 'true'
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', 'localhost').split(',')
```

### Feature Flags
```python
# Future pattern for model-specific features
ENABLE_F5TTS = os.getenv('ENABLE_F5TTS', 'True').lower() == 'true'
ENABLE_E2TTS = os.getenv('ENABLE_E2TTS', 'True').lower() == 'true'
```

## Security Patterns

### File Upload Security
- **File Type Validation**: Check audio file extensions
- **Size Limits**: Prevent large file uploads
- **Temporary Storage**: Auto-cleanup of uploaded files
- **Path Sanitization**: Prevent directory traversal

### Input Validation
- **Parameter Bounds**: Speed (0.1-3.0), duration limits
- **Required Fields**: Text and reference audio validation
- **Data Types**: Type conversion with error handling

## Performance Patterns

### Model Caching
- **Lazy Loading**: Models loaded on first use
- **Memory Management**: Single instance per model type
- **Startup Optimization**: Avoid blocking application start

### Response Optimization
- **Base64 Encoding**: Efficient binary data transfer
- **Streaming**: Future pattern for large audio files
- **Compression**: Future pattern for response compression

## Extensibility Patterns

### Plugin Architecture (Future)
```python
# Pattern for adding new TTS models
class TTSModelPlugin:
    def load_model(self, path: str) -> Any
    def generate_speech(self, *args) -> np.ndarray
```

### Configuration-Driven Models
```python
# Pattern for external model configuration
MODEL_CONFIGS = {
    'F5-TTS': {'checkpoint': 'path/to/f5.ckpt'},
    'E2-TTS': {'checkpoint': 'path/to/e2.ckpt'},
}
```

## Testing Patterns

### Service Testing
- **Mock Models**: Replace actual TTS models with mocks
- **Dummy Audio**: Use sine wave generation for tests
- **Isolation**: Test business logic without Django

### API Testing
- **File Upload Testing**: Mock multipart form data
- **Response Validation**: Check JSON structure and types
- **Error Scenarios**: Test all error conditions 