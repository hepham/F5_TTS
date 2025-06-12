"""
URL Configuration for TTS app
"""

from django.urls import path
from .views import TTSGenerationView, HealthCheckView, ModelsView, GPUStatusView, LanguagesView, DebugModelView

app_name = 'tts'

urlpatterns = [
    path('generate/', TTSGenerationView.as_view(), name='generate'),
    path('health/', HealthCheckView.as_view(), name='health'),
    path('models/', ModelsView.as_view(), name='models'),
    path('languages/', LanguagesView.as_view(), name='languages'),
    path('gpu-status/', GPUStatusView.as_view(), name='gpu-status'),
    path('debug/model/', DebugModelView.as_view(), name='debug-model'),
] 