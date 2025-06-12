# Project Brief: F5-TTS Django REST API

## Project Overview

A Django REST API for text-to-speech generation based on the F5-TTS model, extracted from the infer_gradio.py file but built as a server-side API without UI components and without melspectrogram functionality.

## Core Requirements

### Primary Goals
1. **RESTful API** for text-to-speech generation
2. **No UI components** - API only
3. **No melspectrogram functionality** - audio output only
4. **Based on infer_gradio.py** - leverage existing TTS logic
5. **Support multiple TTS models** (F5-TTS, E2-TTS, Custom)

### Key Features
- Text-to-speech generation with reference audio
- Configurable speech parameters (speed, cross-fade, silence removal)
- Base64 encoded audio output
- File upload support for reference audio
- Auto-transcription of reference audio when ref_text not provided
- Health check and model listing endpoints
- Swagger/OpenAPI documentation

### Technical Constraints
- Django + Django REST Framework
- Python 3.x compatibility
- Cross-platform deployment (Windows development environment)
- No frontend UI
- No melspectrogram visualization/output
- Base64 audio encoding for JSON API responses

## Success Criteria

1. **Functional API** with all endpoints working
2. **Model Integration** - F5-TTS, E2-TTS, and custom model support
3. **Audio Processing** - reference audio upload and speech generation
4. **Documentation** - Complete API documentation with examples
5. **Deployment Ready** - Production-ready configuration options

## Out of Scope

- Web UI or frontend components
- Melspectrogram generation or visualization
- Real-time streaming audio
- Audio editing or post-processing features
- User authentication/authorization (basic API for now)
- Database persistence of generated audio

## Target Architecture

- **Django** as web framework
- **Django REST Framework** for API functionality
- **TTS Service Layer** extracted from infer_gradio.py
- **File Upload Handling** for reference audio
- **Base64 Audio Encoding** for JSON responses
- **Swagger Documentation** for API discoverability 