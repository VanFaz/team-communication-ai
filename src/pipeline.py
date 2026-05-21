from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .audio_utils import convert_to_wav, ensure_dir, safe_stem
from .asr import transcribe_audio
from .metrics import build_interpretation, compute_summary
from .reporting import save_outputs
from .text_models import enrich_utterances, load_models


def _file_path(input_path: str | Path | Any) -> Path:
    if isinstance(input_path, (str, Path)):
        return Path(input_path)
    if hasattr(input_path, "name"):
        return Path(input_path.name)
    raise TypeError("Не удалось определить путь к входному файлу.")


def run_pipeline(
    input_path: str | Path | Any,
    work_dir: str | Path = "output",
    model_size: str = "small",
    language: str = "ru",
    backend: str = "auto",
    enable_diarization: bool = True,
    enable_emotion: bool = True,
    device: str = "auto",
    compute_type: str = "auto",
    batch_size: int = 16,
    hf_token: str | None = None,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-transcribe-diarize",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    unique_run_dir: bool = True,
) -> dict:
    """Run the full communication-analysis pipeline."""
    load_dotenv()
    input_path = _file_path(input_path)
    work_dir = ensure_dir(work_dir)
    audio_dir = ensure_dir(work_dir / "audio")

    stem = safe_stem(input_path)
    if unique_run_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(work_dir / f"{stem}_{timestamp}")
    else:
        run_dir = ensure_dir(work_dir / stem)

    wav_path = convert_to_wav(input_path, audio_dir)
    diarization_model = os.getenv("DIARIZATION_MODEL", diarization_model)

    asr_result = transcribe_audio(
        wav_path,
        model_size=model_size,
        language=language,
        backend=backend,
        batch_size=batch_size,
        device=device,
        compute_type=compute_type,
        enable_diarization=enable_diarization,
        hf_token=hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        openai_model=openai_model,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarization_model=diarization_model,
    )

    models = load_models(enable_emotion=enable_emotion)
    utterances = enrich_utterances(asr_result["utterances"], models)
    summary = compute_summary(utterances, asr_result["duration"])
    interpretation = build_interpretation(summary)

    warnings = []
    warnings.extend(asr_result.get("warnings", []) or [])
    warnings.extend(models.warnings or [])

    result = {
        "meta": {
            "project_title": "Программная реализация прототипа ИИ-системы анализа командной коммуникации",
            "input_file": input_path.name,
            "converted_wav": str(wav_path),
            "language": asr_result["language"],
            "language_probability": asr_result.get("language_probability", 0.0),
            "duration_sec": asr_result["duration"],
            "model_size": model_size,
            "asr_backend": asr_result.get("asr_backend", backend),
            "diarization_enabled": bool(asr_result.get("diarization_enabled")),
            "diarization_model": diarization_model if enable_diarization else None,
            "device": device,
            "compute_type": compute_type,
            "warnings": warnings,
        },
        "summary": summary,
        "interpretation": interpretation,
        "utterances": utterances,
        "full_text": asr_result["full_text"],
    }
    result["artifacts"] = save_outputs(result, run_dir)
    return result
