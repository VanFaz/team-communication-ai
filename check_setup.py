from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from src.asr import _patch_huggingface_hub_compat, _patch_torchaudio_compat


def main() -> int:
    load_dotenv()
    print("Проверка окружения проекта\n")

    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {'OK — ' + ffmpeg if ffmpeg else 'НЕ НАЙДЕН'}")

    hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
    diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1").strip()
    print(f"HF_TOKEN: {'OK' if hf_token else 'НЕ ЗАДАН'}")
    print(f"DIARIZATION_MODEL: {diarization_model or 'не задан'}")
    print(f"PYANNOTE_METRICS_ENABLED: {os.getenv('PYANNOTE_METRICS_ENABLED', '0')}")

    try:
        import torch

        print(f"torch: OK; cuda={torch.cuda.is_available()}")
    except Exception as exc:
        print(f"torch: ошибка импорта — {exc}")

    try:
        import faster_whisper  # noqa: F401

        print("faster-whisper: OK")
    except Exception as exc:
        print(f"faster-whisper: НЕ ГОТОВ — {exc}")

    try:
        _patch_torchaudio_compat()
        _patch_huggingface_hub_compat()
        import pyannote.audio  # noqa: F401

        print("pyannote.audio: OK")
    except Exception as exc:
        print(f"pyannote.audio: НЕ ГОТОВ — {exc}")

    try:
        import whisperx  # noqa: F401

        print("whisperx: OK, можно использовать backend=whisperx")
    except Exception as exc:
        print(f"whisperx: не установлен — это нормально для Windows-режима; используйте backend=faster-whisper. Деталь: {exc}")

    if hf_token and diarization_model:
        print("\nПроверяю доступ к gated-модели pyannote...")
        try:
            _patch_huggingface_hub_compat()
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=diarization_model,
                filename="config.yaml",
                token=hf_token,
            )
            print(f"Доступ к pyannote: OK — {Path(path).name}")
            if diarization_model == "pyannote/speaker-diarization-3.1":
                dependency_path = hf_hub_download(
                    repo_id="pyannote/segmentation-3.0",
                    filename="pytorch_model.bin",
                    token=hf_token,
                )
                print(f"pyannote/segmentation-3.0: OK - {Path(dependency_path).name}")
        except Exception as exc:
            print("Доступ к pyannote: ОШИБКА")
            print(
                "Причина чаще всего в том, что условия модели на Hugging Face не приняты "
                "тем же аккаунтом, для которого создан HF_TOKEN, либо токен создан без права Read."
            )
            print(f"Техническая ошибка: {exc}")

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
