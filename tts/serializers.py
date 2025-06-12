"""
Serializers for TTS API
"""

from rest_framework import serializers


class TTSRequestSerializer(serializers.Serializer):
    """Serializer for TTS generation requests"""
    
    text = serializers.CharField(
        max_length=10000,
        help_text="Text to convert to speech"
    )
    language = serializers.ChoiceField(
        choices=[
            ("en", "English"),
            ("zh", "Chinese"), 
            ("fi", "Finnish"),
            ("fr", "French"),
            ("hi", "Hindi"),
            ("it", "Italian"),
            ("ja", "Japanese"),
            ("ru", "Russian"),
            ("es", "Spanish"),
            ("auto", "Auto-detect")
        ],
        default="auto",
        help_text="Language of the input text. The service will automatically select the appropriate language-specific model: EN/ZH (SWivid/F5-TTS_v1), FI (AsmoKoskinen/F5-TTS_Finnish_Model), FR (RASPIAUDIO/F5-French-MixedSpeakers-reduced), HI (SPRINGLab/F5-Hindi-24KHz), IT (alien79/F5-TTS-italian), JA (Jmica/F5TTS/JA_21999120), RU (hotstone228/F5-TTS-Russian), ES (jpgallegoar/F5-Spanish)."
    )
    model = serializers.ChoiceField(
        choices=[("F5-TTS", "F5-TTS"), ("E2-TTS", "E2-TTS"), ("Custom", "Custom"), ("Auto", "Auto-select by language")],
        default="Auto",
        help_text="TTS model to use. 'Auto' will select model based on language."
    )
    remove_silence = serializers.BooleanField(
        default=False,
        help_text="Whether to remove silences from the output"
    )
    cross_fade_duration = serializers.FloatField(
        default=0.15,
        min_value=0.0,
        max_value=1.0,
        help_text="Duration of cross-fade between audio clips (seconds)"
    )
    speed = serializers.FloatField(
        default=1.0,
        min_value=0.1,
        max_value=3.0,
        help_text="Speed of the generated audio"
    )
    ref_text = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=5000,
        help_text="Reference text (optional, will auto-transcribe if not provided)"
    )
    custom_model_path = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Path to custom model (required if model='Custom')"
    )
    vocab_path = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Path to vocabulary file (for custom models)"
    )

    def validate(self, data):
        """Validate the request data"""
        if data.get('model') == 'Custom' and not data.get('custom_model_path'):
            raise serializers.ValidationError(
                "custom_model_path is required when using Custom model"
            )
        return data


class TTSResponseSerializer(serializers.Serializer):
    """Serializer for TTS generation responses"""
    
    audio_data = serializers.CharField(
        help_text="Base64 encoded audio data"
    )
    sample_rate = serializers.IntegerField(
        help_text="Audio sample rate in Hz"
    )
    ref_text = serializers.CharField(
        help_text="Processed reference text used for generation"
    )
    message = serializers.CharField(
        default="success",
        help_text="Response message"
    )
    model_used = serializers.CharField(
        help_text="TTS model that was used for generation"
    )
    language_detected = serializers.CharField(
        required=False,
        help_text="Detected or specified language"
    )
    processing_info = serializers.DictField(
        required=False,
        help_text="Additional processing information"
    )


class HealthCheckSerializer(serializers.Serializer):
    """Serializer for health check responses"""
    
    status = serializers.CharField()
    message = serializers.CharField()
    available_models = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of available TTS models"
    )
    supported_languages = serializers.ListField(
        child=serializers.CharField(),
        help_text="List of supported languages"
    )


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses"""
    
    error = serializers.CharField(help_text="Error message")
    detail = serializers.CharField(
        required=False,
        help_text="Detailed error information"
    )
    status_code = serializers.IntegerField(help_text="HTTP status code") 