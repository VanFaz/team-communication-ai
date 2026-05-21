from __future__ import annotations

import re
import subprocess
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(path: str | Path) -> str:
    """Return a filesystem-safe stem for output folders/files."""
    stem = Path(path).stem.strip() or "input"
    stem = re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", stem)
    return stem[:80]


def convert_to_wav(input_path: str | Path, output_dir: str | Path) -> Path:
    """Convert any supported audio/video file to mono 16 kHz WAV using FFmpeg.

    WhisperX, pyannote and most diarization pipelines work most predictably on
    mono 16 kHz audio. The original file is not changed.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    output_dir = ensure_dir(output_dir)
    output_path = output_dir / f"{safe_stem(input_path)}_16k.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Не найден ffmpeg. Установите FFmpeg и убедитесь, что команда ffmpeg доступна в PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg не смог обработать файл:\n{stderr}") from exc

    return output_path


def probe_duration(audio_path: str | Path) -> float:
    """Return media duration in seconds using ffprobe; 0.0 if unavailable."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(completed.stdout.decode("utf-8", errors="ignore").strip() or 0.0)
    except Exception:
        return 0.0


def format_ts(seconds: float | int | None) -> str:
    seconds = float(seconds or 0.0)
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
