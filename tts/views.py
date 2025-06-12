from django.shortcuts import render

# Create your views here.

"""
Views for TTS API
"""

import tempfile
import os
import logging
from typing import Dict, Any

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .services import get_tts_service
from .serializers import (
    TTSRequestSerializer,
    TTSResponseSerializer,
    HealthCheckSerializer,
    ErrorResponseSerializer
)

logger = logging.getLogger(__name__)


class TTSGenerationView(APIView):
    """
    API view for text-to-speech generation
    """
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    @swagger_auto_schema(
        operation_description="Generate speech from text using F5-TTS or E2-TTS models with automatic language detection and model selection",
        manual_parameters=[
            openapi.Parameter(
                'ref_audio',
                openapi.IN_FORM,
                description="Reference audio file",
                type=openapi.TYPE_FILE,
                required=True
            ),
            openapi.Parameter(
                'text',
                openapi.IN_FORM,
                description="Text to convert to speech",
                type=openapi.TYPE_STRING,
                required=True
            ),
            openapi.Parameter(
                'language',
                openapi.IN_FORM,
                description="Language of the input text",
                type=openapi.TYPE_STRING,
                enum=['auto', 'en', 'vi', 'zh', 'ja', 'ko', 'fr', 'de', 'es'],
                default='auto'
            ),
            openapi.Parameter(
                'model',
                openapi.IN_FORM,
                description="TTS model to use",
                type=openapi.TYPE_STRING,
                enum=['Auto', 'F5-TTS', 'E2-TTS', 'Custom'],
                default='Auto'
            ),
            openapi.Parameter(
                'remove_silence',
                openapi.IN_FORM,
                description="Whether to remove silences",
                type=openapi.TYPE_BOOLEAN,
                default=False
            ),
            openapi.Parameter(
                'cross_fade_duration',
                openapi.IN_FORM,
                description="Cross-fade duration in seconds",
                type=openapi.TYPE_NUMBER,
                default=0.15
            ),
            openapi.Parameter(
                'speed',
                openapi.IN_FORM,
                description="Speech speed",
                type=openapi.TYPE_NUMBER,
                default=1.0
            ),
            openapi.Parameter(
                'ref_text',
                openapi.IN_FORM,
                description="Reference text (optional)",
                type=openapi.TYPE_STRING,
                required=False
            ),
        ],
        responses={
            200: TTSResponseSerializer,
            400: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        }
    )
    def post(self, request) -> Response:
        """Generate speech from text"""
        temp_audio_path = None
        
        try:
            # Get reference audio file
            ref_audio_file = request.FILES.get('ref_audio')
            if not ref_audio_file:
                return Response(
                    {
                        'error': 'Reference audio file is required',
                        'status_code': 400
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Save uploaded audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                for chunk in ref_audio_file.chunks():
                    temp_audio.write(chunk)
                temp_audio_path = temp_audio.name

            # Prepare request data
            request_data = {
                'text': request.data.get('text', ''),
                'language': request.data.get('language', 'auto'),
                'model': request.data.get('model', 'Auto'),
                'remove_silence': request.data.get('remove_silence', False),
                'cross_fade_duration': float(request.data.get('cross_fade_duration', 0.15)),
                'speed': float(request.data.get('speed', 1.0)),
                'ref_text': request.data.get('ref_text', ''),
                'custom_model_path': request.data.get('custom_model_path', ''),
                'vocab_path': request.data.get('vocab_path', ''),
            }

            # Validate request data
            serializer = TTSRequestSerializer(data=request_data)
            if not serializer.is_valid():
                return Response(
                    {
                        'error': 'Invalid request data',
                        'detail': serializer.errors,
                        'status_code': 400
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

            validated_data = serializer.validated_data

            # Get TTS service and generate speech
            tts_service = get_tts_service()
            
            audio_base64, sample_rate, processed_ref_text, detected_language, processing_info = tts_service.generate_speech_base64(
                ref_audio_path=temp_audio_path,
                ref_text=validated_data.get('ref_text', ''),
                gen_text=validated_data['text'],
                model_name=validated_data['model'],
                language=validated_data['language'],
                remove_silence=validated_data['remove_silence'],
                cross_fade_duration=validated_data['cross_fade_duration'],
                speed=validated_data['speed'],
                custom_model_path=validated_data.get('custom_model_path'),
                vocab_path=validated_data.get('vocab_path')
            )

            # Prepare response
            response_data = {
                'audio_data': audio_base64,
                'sample_rate': sample_rate,
                'ref_text': processed_ref_text,
                'message': 'Speech generation successful',
                'model_used': processing_info.get('final_model', validated_data['model']),
                'language_detected': detected_language,
                'processing_info': processing_info
            }

            logger.info(f"Successfully generated speech using {processing_info.get('final_model', validated_data['model'])} for language {detected_language}")
            return Response(response_data, status=status.HTTP_200_OK)

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return Response(
                {
                    'error': 'Invalid input parameters',
                    'detail': str(e),
                    'status_code': 400
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return Response(
                {
                    'error': 'Error generating speech',
                    'detail': str(e),
                    'status_code': 500
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")


class HealthCheckView(APIView):
    """
    API view for health check
    """

    @swagger_auto_schema(
        operation_description="Check API health and get available models",
        responses={200: HealthCheckSerializer}
    )
    def get(self, request) -> Response:
        """Health check endpoint"""
        try:
            tts_service = get_tts_service()
            available_models = tts_service.get_available_models()
            supported_languages = tts_service.get_supported_languages()
            
            response_data = {
                'status': 'healthy',
                'message': 'TTS API is running with multi-language support',
                'available_models': available_models,
                'supported_languages': supported_languages
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return Response(
                {
                    'error': 'Health check failed',
                    'detail': str(e),
                    'status_code': 500
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelsView(APIView):
    """
    API view for getting available models
    """

    @swagger_auto_schema(
        operation_description="Get list of available TTS models",
        responses={
            200: openapi.Response(
                description="List of available models",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'models': openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(type=openapi.TYPE_STRING)
                        )
                    }
                )
            )
        }
    )
    def get(self, request) -> Response:
        """Get available models"""
        try:
            tts_service = get_tts_service()
            available_models = tts_service.get_available_models()
            
            return Response(
                {'models': available_models},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return Response(
                {
                    'error': 'Error retrieving available models',
                    'detail': str(e),
                    'status_code': 500
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class LanguagesView(APIView):
    """
    API view for language support information
    """

    @swagger_auto_schema(
        operation_description="Get supported languages and language-model mapping",
        responses={
            200: openapi.Response(
                description="Language support information",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'supported_languages': openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Schema(type=openapi.TYPE_STRING)
                        ),
                        'language_model_mapping': openapi.Schema(
                            type=openapi.TYPE_OBJECT,
                            additional_properties=openapi.Schema(type=openapi.TYPE_STRING)
                        ),
                        'description': openapi.Schema(type=openapi.TYPE_STRING)
                    }
                )
            )
        }
    )
    def get(self, request) -> Response:
        """Get language support information"""
        try:
            tts_service = get_tts_service()
            supported_languages = tts_service.get_supported_languages()
            language_model_mapping = tts_service.get_language_model_mapping()
            
            language_descriptions = {
                'auto': 'Automatic language detection',
                'en': 'English',
                'vi': 'Vietnamese (Tiếng Việt)',
                'zh': 'Chinese (中文)',
                'ja': 'Japanese (日本語)',
                'ko': 'Korean (한국어)',
                'fr': 'French (Français)',
                'de': 'German (Deutsch)',
                'es': 'Spanish (Español)'
            }
            
            return Response({
                'supported_languages': supported_languages,
                'language_descriptions': language_descriptions,
                'language_model_mapping': language_model_mapping,
                'description': 'The service automatically selects the best model for each language. Use "auto" for automatic language detection.'
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting language info: {str(e)}")
            return Response(
                {
                    'error': 'Error retrieving language information',
                    'detail': str(e),
                    'status_code': 500
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GPUStatusView(APIView):
    """
    API view for GPU status and memory information
    """
    
    @swagger_auto_schema(
        operation_description="Get GPU status and memory information",
        responses={
            200: openapi.Response(
                description="GPU status information",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'device': openapi.Schema(type=openapi.TYPE_STRING),
                        'gpu_available': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        'gpu_name': openapi.Schema(type=openapi.TYPE_STRING),
                        'memory_allocated': openapi.Schema(type=openapi.TYPE_STRING),
                        'memory_reserved': openapi.Schema(type=openapi.TYPE_STRING),
                        'memory_total': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            )
        }
    )
    def get(self, request) -> Response:
        """Get GPU status"""
        try:
            tts_service = get_tts_service()
            gpu_status = tts_service.get_gpu_status()
            return Response(gpu_status, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error getting GPU status: {str(e)}")
            return Response(
                {
                    'error': 'Error getting GPU status',
                    'detail': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @swagger_auto_schema(
        operation_description="Clear GPU cache to free memory",
        responses={
            200: openapi.Response(
                description="GPU cache cleared successfully",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            )
        }
    )
    def post(self, request) -> Response:
        """Clear GPU cache"""
        try:
            tts_service = get_tts_service()
            tts_service.clear_gpu_cache()
            return Response(
                {'message': 'GPU cache cleared successfully'},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error clearing GPU cache: {str(e)}")
            return Response(
                {
                    'error': 'Failed to clear GPU cache',
                    'detail': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DebugModelView(APIView):
    """Debug view to check local model detection"""
    
    @swagger_auto_schema(
        operation_description="Debug local model detection",
        responses={
            200: openapi.Response(
                description="Debug information for model detection",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'model_name': openapi.Schema(type=openapi.TYPE_STRING),
                        'exists_locally': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        'debug_info': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            )
        }
    )
    def get(self, request) -> Response:
        """Check local model detection for Japanese model"""
        try:
            tts_service = get_tts_service()
            model_name = request.GET.get('model', 'Jmica/F5TTS/JA_21999120')
            
            # Run debug check
            exists = tts_service.debug_model_detection(model_name)
            
            return Response({
                'model_name': model_name,
                'exists_locally': exists,
                'message': 'Debug information logged, check console/logs for details'
            })
            
        except Exception as e:
            logger.error(f"Error in debug check: {str(e)}")
            return Response(
                {
                    'error': 'Debug check failed',
                    'detail': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
