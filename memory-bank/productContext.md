# Product Context: F5-TTS Django REST API

## Why This Project Exists

### Problem Statement
The original `infer_gradio.py` file provides powerful text-to-speech capabilities through F5-TTS and E2-TTS models but is designed as an interactive Gradio application with UI components. There's a need for a server-side API that can be integrated into other applications and services without the overhead of a web UI.

### Business Value
- **API-First Architecture**: Enable integration with mobile apps, web services, and other applications
- **Scalable Deployment**: Run as a microservice without UI dependencies
- **Developer-Friendly**: RESTful API with clear documentation for easy integration
- **Model Flexibility**: Support multiple TTS models with consistent interface

## How It Should Work

### Core User Journey
1. **Upload Reference Audio**: User provides a sample audio file of the target voice
2. **Specify Text**: User provides the text to be converted to speech
3. **Configure Parameters**: User optionally adjusts speed, cross-fade, silence removal
4. **Receive Audio**: API returns base64-encoded audio data with metadata

### Key User Experiences

#### For Developers
- **Simple API Integration**: Single POST endpoint for TTS generation
- **Clear Documentation**: Swagger UI with examples and parameter descriptions
- **Flexible Input**: Support various audio formats for reference
- **Consistent Output**: Standardized JSON response with base64 audio

#### For System Administrators
- **Health Monitoring**: Health check endpoint for service monitoring
- **Model Management**: Endpoint to list available models
- **Configuration**: Environment-based configuration for deployment

### API Design Philosophy

#### RESTful Principles
- **Resource-Based URLs**: `/api/tts/generate/`, `/api/tts/models/`
- **HTTP Methods**: POST for generation, GET for status/info
- **Status Codes**: Proper HTTP status codes for success/failure
- **JSON Responses**: Consistent JSON structure with error handling

#### Developer Experience
- **Auto-Documentation**: Swagger/OpenAPI integration
- **Example Code**: Curl and Python examples in documentation
- **Error Messages**: Clear, actionable error messages
- **Parameter Validation**: Input validation with helpful feedback

## Success Metrics

### Technical Success
- **API Reliability**: 99%+ uptime for health check endpoint
- **Response Time**: < 30 seconds for typical TTS generation
- **Error Handling**: Graceful handling of invalid inputs
- **Documentation**: Complete API documentation with examples

### Integration Success
- **Easy Setup**: < 5 minutes from clone to running API
- **Clear Examples**: Working curl and Python examples
- **Model Support**: All three model types (F5-TTS, E2-TTS, Custom) functional
- **File Handling**: Robust audio file upload and processing

## Target Users

### Primary Users
- **Backend Developers**: Integrating TTS into applications
- **Mobile App Developers**: Adding voice synthesis features
- **API Consumers**: Building voice-enabled services

### Secondary Users
- **DevOps Engineers**: Deploying and monitoring the service
- **Data Scientists**: Using custom trained models
- **QA Engineers**: Testing voice applications

## Quality Standards

### Audio Quality
- **Faithful Voice Cloning**: Generated speech matches reference voice characteristics
- **Clear Output**: No artifacts or distortion in generated audio
- **Configurable Quality**: Speed and cross-fade parameters work as expected

### API Quality
- **Input Validation**: Proper validation of all parameters
- **Error Handling**: Meaningful error messages for all failure cases
- **Response Consistency**: Consistent JSON structure across all endpoints
- **Documentation Accuracy**: Examples that actually work 