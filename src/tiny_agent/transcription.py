import abc
import audioop
import io
import json
import wave
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from src.tiny_agent.models import TinyAgentConfig


@dataclass
class ResampledAudio:
    raw_bytes: bytes
    sample_rate: int


class WhisperClient(abc.ABC):
    """An abstract class for the Whisper Servers."""

    def __init__(self, config: TinyAgentConfig):
        pass

    @abc.abstractmethod
    async def transcribe(self, file: io.BytesIO) -> str:
        """Transcribe the audio to text using Whisper."""
        pass

    @staticmethod
    @abc.abstractmethod
    def resample_audio(raw_bytes: bytes, sample_rate: int) -> ResampledAudio:
        """Resample the audio to the target sample rate which the Whisper server expects"""
        pass


class WhisperOpenAIClient(WhisperClient):

    def __init__(self, config: TinyAgentConfig):
        self.client = AsyncOpenAI(api_key=config.whisper_config.api_key)

    async def transcribe(self, file: io.BytesIO) -> str:
        transcript = await self.client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format="verbose_json",
            language="en",
        )
        return transcript.text

    @staticmethod
    def resample_audio(raw_bytes: bytes, sample_rate: int) -> ResampledAudio:
        # OpenAI Whisper API already resample your bytes so this is just a noop
        return ResampledAudio(raw_bytes, sample_rate)


class WhisperCppClient(WhisperClient):
    # Whisper.cpp server expects 16kHz audio
    _TARGET_SAMPLE_RATE = 16000
    _NON_DATA_FIELDS = {
        "temperature": "0.0",
        "temperature_inc": "0.2",
        "response_format": "json",
    }

    _base_url: str

    def __init__(self, config: TinyAgentConfig):
        self._base_url = f"http://localhost:{config.whisper_config.port}/inference"

    async def transcribe(self, file: io.BytesIO) -> str:
        # Preparing the files dictionary
        files = {"file": ("audio.wav", file, "audio/wav")}

        # Send the request to the Whisper.cpp server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._base_url, files=files, data=WhisperCppClient._NON_DATA_FIELDS
            )

        data = response.json()
        if "text" not in data:
            raise ValueError(
                f"Whisper.cpp response does not contain 'text' key: {json.dumps(data)}"
            )

        return data["text"]

    @staticmethod
    def resample_audio(raw_bytes: bytes, sample_rate: int) -> ResampledAudio:
        if sample_rate == WhisperCppClient._TARGET_SAMPLE_RATE:
            return ResampledAudio(raw_bytes, sample_rate)

        # Resample the audio to 16kHz
        converted, _ = audioop.ratecv(
            raw_bytes, 2, 1, sample_rate, WhisperCppClient._TARGET_SAMPLE_RATE, None
        )
        return ResampledAudio(converted, WhisperCppClient._TARGET_SAMPLE_RATE)


class TranscriptionService:
    """
    This is the main service that deals with transcribing raw audio data to text using Whisper.
    """

    _client: WhisperClient

    def __init__(self, client: WhisperClient):
        self._client = client

    async def transcribe(self, raw_bytes: bytes, sample_rate: int) -> str:
        """
        This method transcribes the audio data to text using Whisper.
        """
        resampled_audio = self._client.resample_audio(raw_bytes, sample_rate)
        raw_bytes, sample_rate = resampled_audio.raw_bytes, resampled_audio.sample_rate

        with io.BytesIO() as memfile:
            # Open a new WAV file in write mode using the in-memory stream
            with wave.open(memfile, "wb") as wav_file:
                # Set the parameters for the WAV file
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit PCM, so 2 bytes per sample
                wav_file.setframerate(sample_rate)

                # Write the PCM data to the WAV file
                wav_file.writeframes(raw_bytes)

            memfile.name = "audio.wav"

            transcript = await self._client.transcribe(memfile)

        return transcript.strip()
