from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from time import time
from typing import Optional

from meerkat.models.provider import ModelBacked, UnsupportedCapability
from meerkat.runtime.logging import RuntimeLogger


class SpeechSink:
    async def speak(self, text: str, stream_time_ms: Optional[int] = None) -> Optional[Path]:
        return None


class NullSpeechSink(SpeechSink):
    pass


class ModelSpeechSink(ModelBacked, SpeechSink):
    """Write each response to an mp3 with the active provider's speech model."""

    def __init__(
        self,
        model: Optional[str] = None,
        voice: str = "alloy",
        output_dir: str = ".meerkat_audio",
        play_audio: bool = False,
        logger: Optional[RuntimeLogger] = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.output_dir = Path(output_dir)
        self.play_audio = play_audio
        self.logger = logger
        self._counter = 0
        self._disabled = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def speak(self, text: str, stream_time_ms: Optional[int] = None) -> Optional[Path]:
        cleaned = " ".join(text.strip().split())
        if not cleaned or self._disabled:
            return None
        self._counter += 1
        path = self.output_dir / f"speech-{int(time() * 1000)}-{self._counter}.mp3"
        if self.logger:
            self.logger.log(stream_time_ms, f"Sending TTS request model={self.model} voice={self.voice}")
        try:
            await asyncio.to_thread(self._write_audio, cleaned, path)
        except UnsupportedCapability as exc:
            self._disabled = True
            if self.logger:
                self.logger.log(stream_time_ms, f"Spoken output disabled: {exc}")
            return None
        if self.logger:
            self.logger.log(stream_time_ms, f"TTS audio ready path={path}")
        if self.play_audio:
            await asyncio.to_thread(_play_audio_file, path)
        return path

    def _write_audio(self, text: str, path: Path) -> None:
        audio = self.provider.synthesize_speech(text, model=self.model, voice=self.voice)
        path.write_bytes(audio)


def _play_audio_file(path: Path) -> None:
    for command in (["afplay", str(path)], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]):
        try:
            subprocess.run(command, check=False)
            return
        except FileNotFoundError:
            continue
