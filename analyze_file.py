from __future__ import annotations

import argparse
import json

from src.pipeline import run_pipeline


def _none_if_empty(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ командной коммуникации по аудио/видео файлу")
    parser.add_argument("input", help="Путь к аудио или видео файлу")
    parser.add_argument("--out", default="output", help="Папка для результатов")
    parser.add_argument("--backend", default="auto", choices=["auto", "whisperx", "faster-whisper", "openai-diarize"], help="ASR backend")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Размер локальной модели Whisper/WhisperX")
    parser.add_argument("--lang", default="ru", help="Код языка аудио, например ru или en")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Устройство для локальных моделей")
    parser.add_argument("--compute-type", default="auto", help="Тип вычислений: auto, int8, float16, float32")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size для WhisperX")
    parser.add_argument("--no-diarization", action="store_true", help="Отключить разделение говорящих")
    parser.add_argument("--min-speakers", type=int, default=None, help="Минимальное число говорящих для diarization")
    parser.add_argument("--max-speakers", type=int, default=None, help="Максимальное число говорящих для diarization")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1", help="pyannote model id для локального diarization")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token для pyannote/WhisperX diarization")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key для backend=openai-diarize")
    parser.add_argument("--openai-model", default="gpt-4o-transcribe-diarize", help="OpenAI ASR модель")
    parser.add_argument("--no-emotion", action="store_true", help="Отключить модель эмоций")
    args = parser.parse_args()

    result = run_pipeline(
        input_path=args.input,
        work_dir=args.out,
        backend=args.backend,
        model_size=args.model,
        language=args.lang,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        enable_diarization=not args.no_diarization,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        diarization_model=args.diarization_model,
        hf_token=_none_if_empty(args.hf_token),
        openai_api_key=_none_if_empty(args.openai_api_key),
        openai_model=args.openai_model,
        enable_emotion=not args.no_emotion,
    )

    print(json.dumps({
        "summary": result["summary"],
        "interpretation": result["interpretation"],
        "artifacts": result["artifacts"],
        "warnings": result["meta"].get("warnings", []),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
