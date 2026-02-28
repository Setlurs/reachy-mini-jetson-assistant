"""Configuration — loads settings.yaml into typed dataclasses."""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import yaml


@dataclass
class LLMConfig:
    model: str = ""
    base_url: str = "http://localhost:8080"
    backend: str = "openai"
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: float = 120.0
    system_prompt: str = "You are a helpful AI assistant."
    system_prompt_no_rag: str = "You are a helpful AI assistant. Answer from your own knowledge."


@dataclass
class STTConfig:
    model: str = "base.en"
    device: str = "cuda"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1


@dataclass
class TTSConfig:
    backend: str = "kokoro"
    voice: str = "af_sarah"
    speed: float = 1.0
    piper_voice: str = "en_US-lessac-medium"
    lang: str = "en-us"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 2
    input_device: Optional[str] = "EMEET"


@dataclass
class VisionConfig:
    camera_device: int = 0
    width: int = 640
    height: int = 480
    jpeg_quality: int = 80
    frames: int = 3
    capture_fps: float = 3.0
    system_prompt: str = (
        "You are a vision assistant on an NVIDIA Jetson device with a live camera. "
        "You receive sequential frames captured while the user was speaking, ordered earliest to latest. "
        "Use differences between frames to understand motion, gestures, and actions. "
        "Answer in one to two sentences. Be direct and concise. "
        "Never use emojis, asterisks, bullet points, markdown, or special formatting."
    )


@dataclass
class RAGConfig:
    enabled: bool = True
    knowledge_dir: str = "./knowledge_base"
    persist_dir: str = "./data/chromadb"
    embedding_backend: str = "llamacpp"
    embedding_model: str = "bge-small-en-v1.5"
    embedding_base_url: str = "http://localhost:8081"
    n_results: int = 3
    min_relevance: float = 0.5
    chunk_size: int = 200
    chunk_overlap: int = 20


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        config = cls()
        if not os.path.exists(config_path):
            return config
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            for section_name, section_obj in [
                ("llm", config.llm), ("stt", config.stt), ("tts", config.tts),
                ("audio", config.audio), ("rag", config.rag), ("vision", config.vision),
            ]:
                for k, v in data.get(section_name, {}).items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
        except Exception as e:
            print(f"Error loading config: {e}")
        return config
