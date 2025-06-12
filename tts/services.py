"""
TTS Service Module

This module contains the core F5-TTS functionality for text-to-speech generation.
Integrated with actual F5-TTS models and inference functions.
"""

import tempfile
import base64
import io
import logging
import sys
import os
import re
import shutil
from typing import Optional, Tuple, Any, Dict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path

# Add the parent directory to the Python path to import F5-TTS modules
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handle downloading and caching of models from Hugging Face"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Hugging Face model configurations
        self.model_configs = {
            "SWivid/F5-TTS_v1": {
                "checkpoint_path": "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "safetensors"
            },
            "AsmoKoskinen/F5-TTS_Finnish_Model": {
                "checkpoint_path": "hf://AsmoKoskinen/F5-TTS_Finnish_Model/model_commonvoice_fi_librivox_fi_vox_populi_fi_20241217/model_last_20241217.safetensors",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "safetensors"
            },
            "RASPIAUDIO/F5-French-MixedSpeakers-reduced": {
                "checkpoint_path": "hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_1374000.pt",
                "vocab_path": "hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "pt"
            },
            "SPRINGLab/F5-Hindi-24KHz": {
                "checkpoint_path": "hf://SPRINGLab/F5-Hindi-24KHz/model_2500000.safetensors",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "safetensors"
            },
            "alien79/F5-TTS-italian": {
                "checkpoint_path": "hf://alien79/F5-TTS-italian/model_159600.safetensors",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "safetensors"
            },
            "Jmica/F5TTS/JA_21999120": {
                "checkpoint_path": "hf://Jmica/F5TTS/JA_21999120/model_21999120.pt",
                "vocab_path": "hf://Jmica/F5TTS/JA_21999120/vocab.txt",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "pt"
            },
            "hotstone228/F5-TTS-Russian": {
                "checkpoint_path": "hf://hotstone228/F5-TTS-Russian/model_last.safetensors",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "safetensors"
            },
            "jpgallegoar/F5-Spanish": {
                "checkpoint_path": "hf://jpgallegoar/F5-Spanish/model_last.pt",
                "vocab_path": "hf://jpgallegoar/F5-Spanish/vocab.txt",
                "model_type": "F5TTS",
                "config": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                "format": "pt"
            }
        }
    
    def get_model_path(self, model_name: str) -> Tuple[Path, Optional[Path]]:
        """Get local paths for a model (checkpoint and optional vocab)"""
        # Convert model name to safe filename
        safe_name = model_name.replace("/", "_").replace(":", "_")
        model_dir = self.models_dir / safe_name
        
        config = self.model_configs.get(model_name, {})
        
        # Determine file extension based on format
        if config.get("format") == "pt":
            checkpoint_file = "model.pt"
        else:
            checkpoint_file = "model.safetensors"
        
        checkpoint_path = model_dir / checkpoint_file
        
        # Check if model has vocab file - try multiple possible vocab file names
        vocab_path = None
        if "vocab_path" in config:
            # Get original filename from config for potential matches
            original_vocab_name = config['vocab_path'].split('/')[-1] if config['vocab_path'] else "vocab.txt"
            
            # Try different vocab file names that might exist locally
            possible_vocab_names = [
                original_vocab_name,                # Original name from HuggingFace (e.g., "vocab_japanese.txt")
                "vocab.txt",                        # Standard name
                "vocab_japanese.txt",               # Japanese specific
                "vocab_french.txt",                 # French specific  
                "vocab_russian.txt",                # Russian specific
                f"vocab_{model_name.split('/')[-1]}.txt",  # Model specific
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_vocab_names = []
            for name in possible_vocab_names:
                if name not in seen:
                    seen.add(name)
                    unique_vocab_names.append(name)
            
            # Check if any of these vocab files exist
            for vocab_name in unique_vocab_names:
                potential_vocab_path = model_dir / vocab_name
                if potential_vocab_path.exists():
                    vocab_path = potential_vocab_path
                    logger.info(f"Found existing vocab file: {vocab_path}")
                    break
            
            # If no existing vocab found, use original name for downloads
            if vocab_path is None:
                vocab_path = model_dir / original_vocab_name
        
        return checkpoint_path, vocab_path
    
    def model_exists_locally(self, model_name: str) -> bool:
        """Check if model exists in local cache"""
        checkpoint_path, vocab_path = self.get_model_path(model_name)
        
        # Check if checkpoint exists
        checkpoint_exists = checkpoint_path.exists()
        
        # Check if vocab exists (if required)
        vocab_exists = True
        if vocab_path is not None:
            vocab_exists = vocab_path.exists()
        
        exists = checkpoint_exists and vocab_exists
        logger.info(f"Model {model_name} local cache check: {'EXISTS' if exists else 'NOT FOUND'}")
        if not exists:
            if not checkpoint_exists:
                logger.info(f"  - Checkpoint missing: {checkpoint_path}")
            if vocab_path and not vocab_exists:
                logger.info(f"  - Vocab missing: {vocab_path}")
        
        return exists
    
    def download_model(self, model_name: str) -> Tuple[str, Optional[str]]:
        """Download model from Hugging Face and cache locally"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        checkpoint_path, vocab_path = self.get_model_path(model_name)
        
        # Create model directory
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model {model_name} from Hugging Face...")
        logger.info(f"Checkpoint source: {config['checkpoint_path']}")
        logger.info(f"Checkpoint destination: {checkpoint_path}")
        
        try:
            # Download checkpoint from Hugging Face using cached_path
            downloaded_checkpoint = cached_path(config['checkpoint_path'])
            
            # Copy checkpoint to our local cache
            shutil.copy2(downloaded_checkpoint, checkpoint_path)
            logger.info(f"Successfully downloaded checkpoint: {checkpoint_path}")
            
            # Download vocab file if needed
            vocab_local_path = None
            if "vocab_path" in config:
                # Extract original filename from Hugging Face path
                original_vocab_name = config['vocab_path'].split('/')[-1]  # e.g., "vocab_japanese.txt"
                vocab_local_path = checkpoint_path.parent / original_vocab_name
                
                logger.info(f"Vocab source: {config['vocab_path']}")
                logger.info(f"Vocab destination: {vocab_local_path}")
                
                downloaded_vocab = cached_path(config['vocab_path'])
                shutil.copy2(downloaded_vocab, vocab_local_path)
                vocab_local_path = str(vocab_local_path)
                logger.info(f"Successfully downloaded vocab with original name: {vocab_local_path}")
            
            logger.info(f"Successfully downloaded and cached {model_name}")
            return str(checkpoint_path), vocab_local_path
            
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {str(e)}")
            raise
    
    def get_or_download_model(self, model_name: str) -> Tuple[str, Optional[str], dict]:
        """Get model paths, downloading if necessary"""
        if self.model_exists_locally(model_name):
            checkpoint_path, vocab_path = self.get_model_path(model_name)
            checkpoint_path_str = str(checkpoint_path)
            vocab_path_str = str(vocab_path) if vocab_path else None
            logger.info(f"Using cached model: {checkpoint_path_str}")
            if vocab_path_str:
                logger.info(f"Using cached vocab: {vocab_path_str}")
        else:
            checkpoint_path_str, vocab_path_str = self.download_model(model_name)
        
        config = self.model_configs[model_name]
        return checkpoint_path_str, vocab_path_str, config


class DummyProgress:
    """Dummy progress class for non-UI context"""
    
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return self
    
    def tqdm(self, iterable, *args, **kwargs):
        """Mimics gr.Progress().tqdm() for API context"""
        return iterable
    
    def update(self, *args, **kwargs):
        """Mimics gr.Progress().update() for API context"""
        pass


class TTSService:
    """
    Service class for handling TTS operations
    Manages model loading, audio generation, and caching
    """
    
    def __init__(self):
        self.vocoder = None
        self.f5tts_model = None
        self.e2tts_model = None
        self.custom_model = None
        self.pre_custom_path = ""
        
        # Initialize model downloader
        self.model_downloader = ModelDownloader()
        
        # Language-specific model mapping based on discovered models
        self.language_model_map = {
            "en": "SWivid/F5-TTS_v1",                           # English & Chinese (official base model)
            "zh": "SWivid/F5-TTS_v1",                           # English & Chinese (official base model)
            "fi": "AsmoKoskinen/F5-TTS_Finnish_Model",          # Finnish
            "fr": "RASPIAUDIO/F5-French-MixedSpeakers-reduced", # French
            "hi": "SPRINGLab/F5-Hindi-24KHz",                   # Hindi
            "it": "alien79/F5-TTS-italian",                     # Italian
            "ja": "Jmica/F5TTS/JA_21999120",                    # Japanese
            "ru": "hotstone228/F5-TTS-Russian",                 # Russian
            "es": "jpgallegoar/F5-Spanish",                     # Spanish
        }
        
        # Cache for loaded language models
        self.language_models = {}
        
        # Supported languages (matching the discovered models)
        self.supported_languages = [
            "en", "zh", "fi", "fr", "hi", "it", "ja", "ru", "es"
        ]
        
        # Setup device - prioritize GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Optimize for GPU if available
        if self.device == "cuda":
            # Enable optimizations for better GPU performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("GPU optimizations enabled")
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the TTS models and vocoder"""
        logger.info("Initializing TTS models...")
        
        try:
            # Load vocoder and move to device
            self.vocoder = load_vocoder()
            if hasattr(self.vocoder, 'to'):
                self.vocoder = self.vocoder.to(self.device)
            logger.info(f"Vocoder loaded successfully on {self.device}")
            
            # Load F5-TTS model and move to device
            self.f5tts_model = self._load_f5tts()
            if hasattr(self.f5tts_model, 'to'):
                self.f5tts_model = self.f5tts_model.to(self.device)
            logger.info(f"F5-TTS model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _load_f5tts(self, ckpt_path: Optional[str] = None) -> Any:
        """Load F5-TTS model"""
        try:
            if ckpt_path is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            
            F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            logger.info(f"Loading F5-TTS model from: {ckpt_path}")
            
            # Use F5-TTS load_model function which handles PyTorch compatibility
            model = load_model(DiT, F5TTS_model_cfg, ckpt_path)
            
            # Move to device and enable optimizations
            if hasattr(model, 'to'):
                model = model.to(self.device)
                if self.device == "cuda":
                    model = model.half()  # Use half precision for faster inference on GPU
                    logger.info("Model moved to GPU with half precision")
            
            logger.info("✅ F5-TTS model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load F5-TTS model: {str(e)}")
            # Return a dummy model or re-raise
            raise
    
    def _load_e2tts(self, ckpt_path: Optional[str] = None) -> Any:
        """Load E2-TTS model"""
        try:
            if ckpt_path is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            
            E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            logger.info(f"Loading E2-TTS model from: {ckpt_path}")
            
            # Use F5-TTS load_model function which handles PyTorch compatibility
            model = load_model(UNetT, E2TTS_model_cfg, ckpt_path)
            
            # Move to device and enable optimizations
            if hasattr(model, 'to'):
                model = model.to(self.device)
                if self.device == "cuda":
                    model = model.half()  # Use half precision for faster inference on GPU
                    logger.info("Model moved to GPU with half precision")
                    
            logger.info("✅ E2-TTS model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load E2-TTS model: {str(e)}")
            # Return a dummy model or re-raise
            raise
    
    def _load_custom(self, ckpt_path: str, vocab_path: str = "", model_cfg: Optional[dict] = None) -> Any:
        """Load custom model"""
        try:
            ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
            if ckpt_path.startswith("hf://"):
                ckpt_path = str(cached_path(ckpt_path))
            if vocab_path.startswith("hf://"):
                vocab_path = str(cached_path(vocab_path))
            if model_cfg is None:
                model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            
            logger.info(f"Loading custom model from: {ckpt_path}")
            if vocab_path:
                logger.info(f"Using vocab file: {vocab_path}")
            
            # Use F5-TTS load_model function which handles PyTorch compatibility
            model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)
            
            # Move to device and enable optimizations
            if hasattr(model, 'to'):
                model = model.to(self.device)
                if self.device == "cuda":
                    model = model.half()  # Use half precision for faster inference on GPU
                    logger.info("Custom model moved to GPU with half precision")
                    
            logger.info("✅ Custom model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {str(e)}")
            # Return a dummy model or re-raise
            raise
    
    def _load_language_model(self, language: str) -> Any:
        """Load language-specific model from Hugging Face with caching"""
        if language in self.language_models:
            logger.info(f"✅ Using cached language model for {language}")
            cached_model = self.language_models[language]
            logger.info(f"Cached model type: {type(cached_model).__name__}")
            return cached_model
        
        if language not in self.language_model_map:
            logger.warning(f"Language '{language}' not supported, falling back to English")
            language = "en"
        
        model_name = self.language_model_map[language]
        logger.info(f"🔄 Loading NEW language model for {language}: {model_name}")
        
        # Debug: Check local model detection
        logger.info(f"🔍 Checking if model exists locally...")
        is_local = self.model_downloader.model_exists_locally(model_name)
        logger.info(f"📁 Local model check result: {is_local}")
        
        # Debug: Show expected paths
        checkpoint_path, vocab_path = self.model_downloader.get_model_path(model_name)
        logger.info(f"📂 Expected checkpoint: {checkpoint_path}")
        logger.info(f"📂 Expected vocab: {vocab_path}")
        logger.info(f"📂 Checkpoint exists: {checkpoint_path.exists()}")
        logger.info(f"📂 Vocab exists: {vocab_path.exists() if vocab_path else 'N/A'}")
        
        try:
            # Get or download the model
            model_path, vocab_path_str, model_config = self.model_downloader.get_or_download_model(model_name)
            logger.info(f"📁 Final model path: {model_path}")
            logger.info(f"📁 Final vocab path: {vocab_path_str}")
            
            # Load the model using the cached/downloaded checkpoint
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading model on device: {device}")
            
            # Create model instance
            if model_config["model_type"] == "F5TTS":
                logger.info(f"Creating DiT model with config: {model_config['config']}")
                model = DiT(**model_config["config"])
            else:
                raise ValueError(f"Unsupported model type: {model_config['model_type']}")
            
            # Load the checkpoint
            logger.info(f"Loading checkpoint from: {model_path}")
            
            # Handle different file formats
            if model_config.get("format") == "pt":
                # Use the load_model function from F5-TTS for .pt files with vocab
                logger.info(f"📁 Loading .pt model with vocab: {vocab_path_str}")
                model = load_model(DiT, model_config["config"], model_path, vocab_file=vocab_path_str or "")
                logger.info("✅ Successfully loaded .pt model with F5-TTS load_model function")
            else:
                # Load .safetensors file directly
                logger.info("📁 Loading .safetensors model directly")
                
                # Fix PyTorch 2.6+ weights_only issue
                try:
                    # Try with safetensors library first (preferred for .safetensors files)
                    try:
                        from safetensors.torch import load_file
                        checkpoint = load_file(model_path, device=str(device))
                        logger.info("✅ Loaded .safetensors using safetensors library")
                    except ImportError:
                        # Fallback to torch.load with weights_only=False for PyTorch 2.6+
                        logger.info("safetensors library not available, using torch.load with weights_only=False")
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        logger.info("✅ Loaded .safetensors using torch.load")
                except Exception as e:
                    logger.warning(f"Failed to load with preferred method: {e}")
                    # Final fallback - try torch.load with weights_only=False
                    logger.info("Trying final fallback with torch.load weights_only=False")
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                # Load state dict
                if 'ema_model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['ema_model_state_dict'])
                    logger.info("✅ Loaded ema_model_state_dict")
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✅ Loaded model_state_dict")
                else:
                    # Direct state dict
                    model.load_state_dict(checkpoint)
                    logger.info("✅ Loaded direct state dict")
            
            model.to(device)
            model.eval()
            logger.info(f"✅ Model moved to {device} and set to eval mode")
            
            # Cache the loaded model
            self.language_models[language] = model
            logger.info(f"🎯 Successfully loaded and cached language model for {language}: {model_name}")
            logger.info(f"Model type in cache: {type(model).__name__}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load language model for {language}: {str(e)}")
            # Fallback to English model if we're not already trying English
            if language != "en":
                logger.info("Falling back to English model")
                return self._load_language_model("en")
            
            # If English also fails, use the dummy model
            logger.warning("All model loading failed, using dummy model")
            return self._load_f5tts()  # This will return the dummy model
    
    def debug_model_detection(self, model_name: str = "Jmica/F5TTS/JA_21999120"):
        """Debug method to check local model detection"""
        logger.info(f"🔍 === Debug Model Detection for: {model_name} ===")
        
        # Check model config
        if model_name in self.model_downloader.model_configs:
            config = self.model_downloader.model_configs[model_name]
            logger.info(f"📋 Model config found: {config}")
        else:
            logger.error(f"❌ Model config not found for: {model_name}")
            return
        
        # Check paths
        checkpoint_path, vocab_path = self.model_downloader.get_model_path(model_name)
        logger.info(f"📂 Expected checkpoint: {checkpoint_path}")
        logger.info(f"📂 Expected vocab: {vocab_path}")
        
        # Check if files exist
        logger.info(f"📁 Checkpoint exists: {checkpoint_path.exists()}")
        logger.info(f"📁 Vocab exists: {vocab_path.exists() if vocab_path else 'N/A'}")
        
        # Check model_exists_locally
        exists = self.model_downloader.model_exists_locally(model_name)
        logger.info(f"🎯 model_exists_locally result: {exists}")
        
        # List actual files in directory
        model_dir = checkpoint_path.parent
        if model_dir.exists():
            logger.info(f"📂 Actual files in {model_dir}:")
            for file in model_dir.iterdir():
                logger.info(f"   - {file.name} ({file.stat().st_size / (1024*1024*1024):.1f}GB)")
        else:
            logger.info(f"📂 Directory does not exist: {model_dir}")
        
        logger.info("🔍 === End Debug ===")
        return exists
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from text using simple heuristics
        Returns language code or 'en' as default
        """
        # Simple language detection based on character patterns
        
        # Chinese - check for Chinese characters
        chinese_chars = r'[\u4e00-\u9fff]'
        if re.search(chinese_chars, text):
            return 'zh'
        
        # Japanese - check for Hiragana, Katakana
        japanese_chars = r'[\u3040-\u309f\u30a0-\u30ff]'
        if re.search(japanese_chars, text):
            return 'ja'
        
        # Russian - check for Cyrillic characters
        russian_chars = r'[\u0400-\u04ff]'
        if re.search(russian_chars, text.lower()):
            return 'ru'
        
        # Hindi - check for Devanagari script
        hindi_chars = r'[\u0900-\u097f]'
        if re.search(hindi_chars, text):
            return 'hi'
        
        # Finnish - check for Finnish-specific characters
        finnish_chars = r'[äöåÄÖÅ]'
        if re.search(finnish_chars, text.lower()):
            return 'fi'
        
        # Italian - check for Italian-specific characters and patterns
        italian_chars = r'[àèéìíîòóùú]'
        italian_patterns = r'\b(il|la|le|gli|un|una|di|da|in|con|su|per|tra|fra|è|che|non|si|ha|ho|hai|abbiamo|avete|hanno)\b'
        if re.search(italian_chars, text.lower()) or re.search(italian_patterns, text.lower()):
            return 'it'
        
        # French - check for French-specific characters and patterns
        french_chars = r'[àâäæçéèêëïîôùûüÿñ]'
        french_patterns = r'\b(le|la|les|un|une|de|du|des|et|est|avec|pour|par|sur|dans|ou|où|que|qui|ne|pas|tout|tous|être|avoir|faire)\b'
        if re.search(french_chars, text.lower()) or re.search(french_patterns, text.lower()):
            return 'fr'
        
        # Spanish - check for Spanish-specific characters and patterns
        spanish_chars = r'[ñáéíóúüÑÁÉÍÓÚÜ]'
        spanish_patterns = r'\b(el|la|los|las|un|una|de|del|y|en|con|por|para|que|es|son|está|están|ser|estar|haber|tener)\b'
        if re.search(spanish_chars, text.lower()) or re.search(spanish_patterns, text.lower()):
            return 'es'
        
        # Default to English if no pattern matches
        return 'en'
    
    def select_model_by_language(self, language: str, preferred_model: str = "Auto") -> str:
        """
        Select appropriate language-specific model based on language
        """
        if preferred_model != "Auto" and preferred_model != "":
            return preferred_model
        
        # Return the language-specific model name
        model_name = self.language_model_map.get(language, "SWivid/F5-TTS_v1")
        logger.info(f"Selected model for language '{language}': {model_name}")
        return model_name
    
    def _get_model(self, model_name: str, custom_path: Optional[str] = None, vocab_path: Optional[str] = None, language: str = "en") -> Any:
        """Get the appropriate model based on model name and language"""
        logger.info(f"🎯 _get_model called with model_name: {model_name}, language: {language}")
        
        # Handle legacy model names with language consideration
        if model_name == "F5-TTS":
            # Check if we need a language-specific model
            print("hehe debug language:", language)
            if language in self.language_model_map and language not in ["en", "zh"]:
                # Use language-specific model instead of generic F5-TTS
                language_model = self.language_model_map[language]
                logger.info(f"🌍 Switching from F5-TTS to language-specific model for {language}: {language_model}")
                return self._load_language_model(language)
            else:
                # Use default F5-TTS for English/Chinese or unsupported languages
                logger.info("📦 Loading legacy F5-TTS model (for en/zh or fallback)")
                if self.f5tts_model is None:
                    self.f5tts_model = self._load_f5tts()
                return self.f5tts_model
        elif model_name == "E2-TTS":
            logger.info("📦 Loading legacy E2-TTS model")
            if self.e2tts_model is None:
                self.e2tts_model = self._load_e2tts()
            return self.e2tts_model
        elif model_name == "Custom" and custom_path:
            logger.info(f"📦 Loading custom model from: {custom_path}")
            if self.pre_custom_path != custom_path:
                self.custom_model = self._load_custom(custom_path, vocab_path or "")
                self.pre_custom_path = custom_path
            return self.custom_model
        # Handle language-specific models
        elif model_name in self.language_model_map.values():
            logger.info(f"🌍 Loading language-specific model: {model_name}")
            # Find which language this model belongs to
            for lang, model in self.language_model_map.items():
                if model == model_name:
                    logger.info(f"🎯 Found model {model_name} for language: {lang}")
                    return self._load_language_model(lang)
            raise ValueError(f"Language model not found: {model_name}")
        else:
            logger.error(f"❌ Unsupported model: {model_name}")
            raise ValueError(f"Unsupported model: {model_name}")
    
    def generate_speech(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        model_name: str = "Auto",
        language: str = "auto",
        remove_silence: bool = False,
        cross_fade_duration: float = 0.15,
        speed: float = 1.0,
        custom_model_path: Optional[str] = None,
        vocab_path: Optional[str] = None
    ) -> Tuple[int, np.ndarray, str, str, Dict]:
        """
        Generate speech from text using the specified TTS model
        
        Args:
            ref_audio_path: Path to reference audio file
            ref_text: Reference text (will be auto-transcribed if empty)
            gen_text: Text to generate speech for
            model_name: TTS model to use ("F5-TTS", "E2-TTS", "Auto", or "Custom")
            language: Language code ("auto", "en", "vi", "zh", etc.)
            remove_silence: Whether to remove silences from the output
            cross_fade_duration: Duration of cross-fade between audio clips
            speed: Speed of the generated audio
            custom_model_path: Path to custom model (if using Custom model)
            vocab_path: Path to vocabulary file (if using Custom model)
        
        Returns:
            Tuple of (sample_rate, audio_data, processed_ref_text, detected_language, processing_info)
        """
        logger.info(f"Generating speech with model: {model_name}, language: {language} on device: {self.device}")
        logger.info(f"Text to generate: {gen_text}")
        
        # Use autocast for GPU to improve performance
        autocast_context = torch.cuda.amp.autocast() if self.device == "cuda" else torch.no_grad()
        
        processing_info = {}
        
        try:
            # Validate inputs
            if not gen_text or not gen_text.strip():
                raise ValueError("Text cannot be empty")
            
            gen_text = str(gen_text).strip()
            ref_text = str(ref_text).strip() if ref_text else ""
            
            # Detect language if auto
            detected_language = language
            if language == "auto":
                detected_language = self.detect_language(gen_text)
                processing_info["language_detection"] = "automatic"
            else:
                processing_info["language_detection"] = "manual"
            
            logger.info(f"Detected/Selected language: {detected_language}")
            
            # Select model based on language if model is Auto
            final_model_name = model_name
            if model_name == "Auto":
                final_model_name = self.select_model_by_language(detected_language, model_name)
                processing_info["model_selection"] = "automatic"
                logger.info(f"Auto-selected model: {final_model_name} for language: {detected_language}")
            else:
                processing_info["model_selection"] = "manual"
            
            processing_info["final_model"] = final_model_name
            processing_info["detected_language"] = detected_language
            
            # Preprocess reference audio and text
            logger.info("Preprocessing reference audio and text...")
            processed_ref_audio, processed_ref_text = preprocess_ref_audio_text(
                ref_audio_path, 
                ref_text, 
                show_info=lambda x: logger.info(f"Preprocessing: {x}")
            )
            
            # Move processed audio to the correct device if it's a tensor
            if torch.is_tensor(processed_ref_audio):
                processed_ref_audio = processed_ref_audio.to(self.device)
                logger.info("Preprocessed audio moved to device")
            
            # Get the appropriate model object using the updated _get_model method
            logger.info(f"Getting model object for: {final_model_name}")
            ema_model = self._get_model(final_model_name, custom_model_path, vocab_path, detected_language)
            
            # Log which model is actually being used
            logger.info(f"Using model: {type(ema_model).__name__} for language: {detected_language}")
            
            # Ensure model is on the correct device
            if hasattr(ema_model, 'to'):
                ema_model = ema_model.to(self.device)
                logger.info(f"Model moved to device: {self.device}")
            
            # Generate speech using the infer_process function with autocast for GPU optimization
            logger.info(f"Starting speech generation with final model: {final_model_name}")
            with autocast_context:
                final_wave, final_sample_rate, combined_spectrogram = infer_process(
                    processed_ref_audio,
                    processed_ref_text,
                    gen_text,
                    ema_model,
                    self.vocoder,
                    cross_fade_duration=cross_fade_duration,
                    speed=speed,
                    show_info=lambda x: logger.info(f"Generation: {x}"),
                    progress=DummyProgress(),
                )
            
            # Ensure output is on CPU for further processing
            if torch.is_tensor(final_wave):
                final_wave = final_wave.cpu()
            
            # Remove silence if requested
            if remove_silence:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    sf.write(f.name, final_wave, final_sample_rate)
                    remove_silence_for_generated_wav(f.name)
                    final_wave, _ = torchaudio.load(f.name)
                    os.unlink(f.name)  # Clean up temp file
                final_wave = final_wave.squeeze().cpu().numpy()
            
            # Convert to numpy if it's still a tensor
            if torch.is_tensor(final_wave):
                final_wave = final_wave.numpy()
            
            logger.info("Speech generation completed successfully")
            return final_sample_rate, final_wave, processed_ref_text, detected_language, processing_info
            
        except Exception as e:
            logger.error(f"Error in speech generation: {str(e)}")
            raise
    
    def generate_speech_base64(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        model_name: str = "Auto",
        language: str = "auto",
        remove_silence: bool = False,
        cross_fade_duration: float = 0.15,
        speed: float = 1.0,
        custom_model_path: Optional[str] = None,
        vocab_path: Optional[str] = None
    ) -> Tuple[str, int, str, str, Dict]:
        """
        Generate speech and return as base64 encoded audio
        
        Returns:
            Tuple of (base64_audio, sample_rate, processed_ref_text, detected_language, processing_info)
        """
        sample_rate, audio_data, processed_ref_text, detected_language, processing_info = self.generate_speech(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            model_name=model_name,
            language=language,
            remove_silence=remove_silence,
            cross_fade_duration=cross_fade_duration,
            speed=speed,
            custom_model_path=custom_model_path,
            vocab_path=vocab_path
        )
        
        # Convert audio to base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return audio_base64, sample_rate, processed_ref_text, detected_language, processing_info
    
    def get_gpu_status(self) -> dict:
        """Get GPU status and memory information"""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_info = {
                "device": self.device,
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            }
        else:
            gpu_info = {
                "device": self.device,
                "gpu_available": False,
                "message": "CUDA not available, using CPU"
            }
        return gpu_info
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    def get_available_models(self) -> list:
        """Get list of available TTS models including language-specific models"""
        models = [
            {
                "name": "Auto", 
                "description": "Auto-select model based on language",
                "loaded": True,
                "device": self.device
            },
            {
                "name": "F5-TTS", 
                "description": "F5 Text-to-Speech model (Legacy)",
                "loaded": self.f5tts_model is not None,
                "device": self.device
            },
            {
                "name": "E2-TTS", 
                "description": "E2 Text-to-Speech model (Legacy)",
                "loaded": self.e2tts_model is not None,
                "device": self.device
            },
            {
                "name": "Custom", 
                "description": "Custom model with user-provided path",
                "loaded": self.custom_model is not None,
                "device": self.device
            }
        ]
        
        # Add language-specific models
        language_models = [
            {
                "name": "SWivid/F5-TTS_v1",
                "description": "English & Chinese - Official F5-TTS base model",
                "languages": ["en", "zh"],
                "format": "safetensors",
                "has_vocab": False,
                "loaded": "en" in self.language_models or "zh" in self.language_models,
                "device": self.device
            },
            {
                "name": "AsmoKoskinen/F5-TTS_Finnish_Model",
                "description": "Finnish language model",
                "languages": ["fi"],
                "format": "safetensors",
                "has_vocab": False,
                "loaded": "fi" in self.language_models,
                "device": self.device
            },
            {
                "name": "RASPIAUDIO/F5-French-MixedSpeakers-reduced",
                "description": "French language model with mixed speakers",
                "languages": ["fr"],
                "format": "pt",
                "has_vocab": True,
                "loaded": "fr" in self.language_models,
                "device": self.device
            },
            {
                "name": "SPRINGLab/F5-Hindi-24KHz",
                "description": "Hindi language model (24kHz)",
                "languages": ["hi"],
                "format": "safetensors",
                "has_vocab": False,
                "loaded": "hi" in self.language_models,
                "device": self.device
            },
            {
                "name": "alien79/F5-TTS-italian",
                "description": "Italian language model",
                "languages": ["it"],
                "format": "safetensors",
                "has_vocab": False,
                "loaded": "it" in self.language_models,
                "device": self.device
            },
            {
                "name": "Jmica/F5TTS/JA_21999120",
                "description": "Japanese language model",
                "languages": ["ja"],
                "format": "pt",
                "has_vocab": True,
                "loaded": "ja" in self.language_models,
                "device": self.device
            },
            {
                "name": "hotstone228/F5-TTS-Russian",
                "description": "Russian language model",
                "languages": ["ru"],
                "format": "safetensors",
                "has_vocab": False,
                "loaded": "ru" in self.language_models,
                "device": self.device
            },
            {
                "name": "jpgallegoar/F5-Spanish",
                "description": "Spanish language model",
                "languages": ["es"],
                "format": "safetensors",
                "has_vocab": False,
                "loaded": "es" in self.language_models,
                "device": self.device
            }
        ]
        
        return models + language_models
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return self.supported_languages + ["auto"]
    
    def get_language_model_mapping(self) -> Dict[str, str]:
        """Get the language to model mapping"""
        return self.language_model_map.copy()


# Global TTS service instance
_tts_service_instance = None


def get_tts_service() -> TTSService:
    """Get or create the global TTS service instance"""
    global _tts_service_instance
    if _tts_service_instance is None:
        _tts_service_instance = TTSService()
    return _tts_service_instance 