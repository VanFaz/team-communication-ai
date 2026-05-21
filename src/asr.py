from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from .audio_utils import probe_duration

# На Windows pyannote.audio 4.x может падать в telemetry/get_duration из-за torchcodec:
# NameError: name 'AudioDecoder' is not defined. Отключаем telemetry до импорта pyannote.
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_device(device: str | None) -> str:
    if not device or device == "auto":
        return _torch_device()
    return device


def _resolve_compute_type(device: str, compute_type: str | None) -> str:
    if compute_type and compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _segment_speaker_from_words(words: list[dict[str, Any]] | None) -> str | None:
    if not words:
        return None
    weights: Counter[str] = Counter()
    for word in words:
        speaker = word.get("speaker")
        if not speaker:
            continue
        start = _float(word.get("start"))
        end = _float(word.get("end"))
        weights[str(speaker)] += max(0.01, end - start)
    return weights.most_common(1)[0][0] if weights else None


def _words_to_plain(words: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    plain: list[dict[str, Any]] = []
    for word in words or []:
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        item = {
            "word": text,
            "start": round(_float(word.get("start")), 3),
            "end": round(_float(word.get("end")), 3),
        }
        if word.get("speaker"):
            item["speaker"] = str(word["speaker"])
        plain.append(item)
    return plain


def _patch_torchaudio_compat() -> None:
    """Restore torchaudio 2.11 symbols still expected by pyannote.audio 3.x."""
    try:
        import soundfile as sf
        import torch
        import torchaudio
        from typing import NamedTuple
    except Exception:
        return

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData(NamedTuple):
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int
            encoding: str

        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(torchaudio, "info"):
        def info(filepath, backend=None):
            metadata = sf.info(filepath)
            return torchaudio.AudioMetaData(
                sample_rate=int(metadata.samplerate),
                num_frames=int(metadata.frames),
                num_channels=int(metadata.channels),
                bits_per_sample=0,
                encoding=str(metadata.subtype or metadata.format or "UNKNOWN"),
            )

        torchaudio.info = info

    if not hasattr(torchaudio, "load"):
        def load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, backend=None):
            frames = num_frames if num_frames is not None and num_frames > 0 else -1
            data, sample_rate = sf.read(
                filepath,
                start=max(0, int(frame_offset or 0)),
                frames=frames,
                dtype="float32",
                always_2d=True,
            )
            waveform = torch.from_numpy(data.T.copy() if channels_first else data.copy())
            return waveform, int(sample_rate)

        torchaudio.load = load


def _patch_huggingface_hub_compat() -> None:
    """Map legacy use_auth_token to token for newer huggingface_hub releases."""
    try:
        import inspect
        import sys

        import huggingface_hub
    except Exception:
        return

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_team_comm_compat", False):
        return

    parameters = inspect.signature(original).parameters
    if "use_auth_token" in parameters or "token" not in parameters:
        return

    def hf_hub_download(*args, use_auth_token=None, token=None, **kwargs):
        if token is None and use_auth_token is not None:
            token = use_auth_token
        return original(*args, token=token, **kwargs)

    hf_hub_download._team_comm_compat = True
    huggingface_hub.hf_hub_download = hf_hub_download

    for module_name in ("pyannote.audio.core.pipeline", "pyannote.audio.core.model"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "hf_hub_download"):
            module.hf_hub_download = hf_hub_download


def _patch_torch_serialization_compat() -> None:
    """Allow trusted pyannote checkpoints to load on PyTorch 2.6+."""
    try:
        import torch
        from torch.torch_version import TorchVersion
    except Exception:
        return

    safe_globals = [TorchVersion]
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        safe_globals.extend([Problem, Resolution, Specifications])
    except Exception:
        pass

    try:
        torch.serialization.add_safe_globals(safe_globals)
    except Exception:
        pass


def _patch_speechbrain_lazy_compat() -> None:
    """Avoid importing optional speechbrain integrations during inspect.stack."""
    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:
        return

    if getattr(LazyModule, "_team_comm_compat", False):
        return

    original_getattr = LazyModule.__getattr__

    def __getattr__(self, name):
        if name == "__file__":
            raise AttributeError(name)
        return original_getattr(self, name)

    LazyModule.__getattr__ = __getattr__
    LazyModule._team_comm_compat = True



def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


@lru_cache(maxsize=2)
def _get_pyannote_pipeline(model_id: str, token: str, device: str):
    # Важно: отключить telemetry до загрузки Pipeline. В pyannote.audio 4.x
    # telemetry может вызвать Audio().get_duration(file), а на Windows это иногда
    # падает из-за torchcodec: NameError: name 'AudioDecoder' is not defined.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    _patch_torchaudio_compat()
    _patch_huggingface_hub_compat()
    _patch_torch_serialization_compat()
    _patch_speechbrain_lazy_compat()

    from pyannote.audio import Pipeline

    try:
        from pyannote.audio.telemetry import set_telemetry_metrics

        set_telemetry_metrics(False)
    except Exception:
        pass

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

    if device == "cuda":
        try:
            import torch

            pipeline.to(torch.device("cuda"))
        except Exception:
            pass
    return pipeline


def _friendly_diarization_error(exc: Exception, model_id: str) -> str:
    message = str(exc)
    lowered = message.lower()
    if "gated" in lowered or "403" in lowered or "restricted" in lowered or "authorized list" in lowered:
        return (
            "Diarization не выполнен: нет доступа к gated-модели pyannote. "
            f"Модель: {model_id}. Нужно открыть страницу модели на Hugging Face, принять условия доступа "
            "тем же аккаунтом, которым создан HF_TOKEN, затем перезапустить приложение. "
            "Также проверьте, что токен имеет право Read и записан в .env как HF_TOKEN. "
            f"Техническая ошибка: {message}"
        )
    if "401" in lowered or "unauthorized" in lowered or "invalid token" in lowered:
        return (
            "Diarization не выполнен: Hugging Face token не принят. "
            "Проверьте HF_TOKEN в .env, права Read и отсутствие лишних пробелов/кавычек. "
            f"Техническая ошибка: {message}"
        )
    if "nonetype" in lowered and "eval" in lowered:
        return (
            "Diarization не выполнен: pyannote не смог загрузить одну из gated-зависимостей модели. "
            f"Модель: {model_id}. Для pyannote/speaker-diarization-3.1 также нужно принять условия доступа "
            "к pyannote/segmentation-3.0 на Hugging Face тем же аккаунтом, чей HF_TOKEN указан в .env. "
            f"Техническая ошибка: {message}"
        )
    if "audiodecoder" in lowered or "torchcodec" in lowered or "libtorchcodec" in lowered:
        return (
            "Diarization не выполнен: проблема аудио-декодера pyannote/torchcodec. "
            "Проверьте, что в .env есть PYANNOTE_METRICS_ENABLED=0 и используется обновлённый src/asr.py. "
            "В этой версии проекта telemetry pyannote отключается, а аудио передаётся как waveform через soundfile. "
            f"Техническая ошибка: {message}"
        )
    return f"Diarization не выполнен: {message}"


def _load_audio_for_pyannote(audio_path: str | Path) -> dict[str, Any]:
    """Load already-converted WAV manually and pass waveform to pyannote.

    This avoids pyannote/torchcodec audio decoding on Windows, where errors like
    `name 'AudioDecoder' is not defined` or `Could not load libtorchcodec` are common.
    The project converts input media to mono 16 kHz WAV before this step, but the
    loader also handles stereo defensively.
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    # soundfile returns [samples, channels], pyannote expects [channels, samples].
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)
    waveform = torch.from_numpy(data.T.copy())
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _run_pyannote_diarization(
    audio_path: str | Path,
    hf_token: str,
    device: str,
    diarization_model: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[dict[str, Any]]:
    # Дополнительная защита: telemetry pyannote может пытаться вычислить duration
    # через AudioDecoder даже когда мы передаём waveform. Отключаем перед вызовом.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

    pipeline = _get_pyannote_pipeline(diarization_model, hf_token, device)

    # Если пользователь точно указал число говорящих, pyannote лучше передавать
    # num_speakers, а не min=max: это уменьшает риск склейки нескольких людей.
    if min_speakers is not None and max_speakers is not None and min_speakers == max_speakers:
        kwargs = {"num_speakers": min_speakers}
    else:
        kwargs = {"min_speakers": min_speakers, "max_speakers": max_speakers}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Do not pass a file path here. On Windows, pyannote.audio>=4 may delegate
    # decoding to torchcodec, and incompatible torch/torchcodec/FFmpeg builds
    # can crash before diarization starts. Passing waveform bypasses that layer.
    diarization_input = _load_audio_for_pyannote(audio_path)
    diarization_input["uri"] = Path(audio_path).stem
    diarization = pipeline(diarization_input, **kwargs)

    turns: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    return turns


def _best_speaker_for_interval(start: float, end: float, diarization_turns: list[dict[str, Any]]) -> str | None:
    weights: Counter[str] = Counter()
    for turn in diarization_turns:
        speaker = str(turn.get("speaker"))
        weights[speaker] += _overlap(start, end, _float(turn.get("start")), _float(turn.get("end")))
    if not weights:
        return None
    speaker, weight = weights.most_common(1)[0]
    return speaker if weight > 0 else None


def _assign_pyannote_speakers(result: dict[str, Any], diarization_turns: list[dict[str, Any]]) -> dict[str, Any]:
    for seg in result.get("segments", []) or []:
        words = seg.get("words") or []
        for word in words:
            start = _float(word.get("start"))
            end = _float(word.get("end"), start)
            speaker = _best_speaker_for_interval(start, end, diarization_turns)
            if speaker:
                word["speaker"] = speaker
        seg_start = _float(seg.get("start"))
        seg_end = _float(seg.get("end"), seg_start)
        seg["speaker"] = _segment_speaker_from_words(words) or _best_speaker_for_interval(seg_start, seg_end, diarization_turns) or seg.get("speaker")
    return result

def _normalize_segments(segments: list[dict[str, Any]], duration: float, language: str) -> dict[str, Any]:
    utterances: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    for idx, seg in enumerate(segments, start=1):
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        words = _words_to_plain(seg.get("words"))
        speaker = seg.get("speaker") or _segment_speaker_from_words(words) or "SPEAKER_00"
        item = {
            "id": idx,
            "speaker": str(speaker),
            "start": round(_float(seg.get("start")), 3),
            "end": round(_float(seg.get("end")), 3),
            "text": text,
        }
        if words:
            item["words"] = words
        for key in ("avg_logprob", "no_speech_prob"):
            if key in seg:
                item[key] = round(_float(seg.get(key)), 4)
        utterances.append(item)
        full_text_parts.append(text)

    return {
        "language": language,
        "language_probability": 0.0,
        "duration": round(float(duration or 0.0), 3),
        "utterances": utterances,
        "full_text": " ".join(full_text_parts).strip(),
        "asr_backend": "normalized",
        "diarization_enabled": any(u.get("speaker") != "SPEAKER_00" for u in utterances),
    }


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _transcribe_faster_whisper(
    audio_path: str | Path,
    model_size: str,
    language: str,
    beam_size: int,
    vad_filter: bool,
    compute_type: str,
    device: str,
    enable_diarization: bool = True,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarization_model: str = "pyannote/speaker-diarization-3.1",
) -> dict[str, Any]:
    """Transcribe with faster-whisper and optionally assign pyannote speakers.

    This Windows-stable path avoids the WhisperX -> pyannote.audio>=4 -> torchcodec
    dependency chain. It still provides word timestamps from faster-whisper and speaker
    labels from pyannote.audio 3.x/4.x when diarization is available.
    """
    audio_path = Path(audio_path)
    model = _get_faster_whisper_model(model_size, device, compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=True,
    )

    segment_dicts: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments, start=1):
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        words = []
        for word in getattr(seg, "words", None) or []:
            w_text = (getattr(word, "word", "") or "").strip()
            if w_text:
                words.append(
                    {
                        "word": w_text,
                        "start": round(_float(getattr(word, "start", None)), 3),
                        "end": round(_float(getattr(word, "end", None)), 3),
                    }
                )
        item: dict[str, Any] = {
            "id": idx,
            "speaker": "SPEAKER_00",
            "start": round(_float(getattr(seg, "start", None)), 3),
            "end": round(_float(getattr(seg, "end", None)), 3),
            "text": text,
            "avg_logprob": round(_float(getattr(seg, "avg_logprob", None)), 4),
            "no_speech_prob": round(_float(getattr(seg, "no_speech_prob", None)), 4),
        }
        if words:
            item["words"] = words
        segment_dicts.append(item)

    detected_language = getattr(info, "language", language) or language
    duration = _float(getattr(info, "duration", None), probe_duration(audio_path))
    result: dict[str, Any] = {
        "segments": segment_dicts,
        "language": detected_language,
        "language_probability": _float(getattr(info, "language_probability", 0.0)),
        "warnings": [],
    }

    diarization_enabled = False
    if enable_diarization:
        if not hf_token:
            result["warnings"].append(
                "Diarization пропущен: не задан HF_TOKEN. Укажите токен Hugging Face в .env или в интерфейсе."
            )
        else:
            try:
                diarization_turns = _run_pyannote_diarization(
                    audio_path=audio_path,
                    hf_token=hf_token,
                    device=device,
                    diarization_model=diarization_model,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                result = _assign_pyannote_speakers(result, diarization_turns)
                diarization_enabled = bool(diarization_turns)
                result["diarization_model"] = diarization_model
            except Exception as exc:
                result["warnings"].append(_friendly_diarization_error(exc, diarization_model))

    normalized = _normalize_segments(result.get("segments", []), duration, detected_language)
    normalized["language_probability"] = result["language_probability"]
    normalized["asr_backend"] = "faster-whisper+pyannote" if diarization_enabled else "faster-whisper"
    normalized["diarization_enabled"] = diarization_enabled and any(
        u.get("speaker") != "SPEAKER_00" for u in normalized["utterances"]
    )
    normalized["warnings"] = result.get("warnings", [])
    return normalized


@lru_cache(maxsize=2)
def _get_whisperx_model(model_size: str, device: str, compute_type: str, language: str | None):
    _patch_torchaudio_compat()
    _patch_huggingface_hub_compat()
    _patch_torch_serialization_compat()
    import whisperx

    kwargs = {"compute_type": compute_type}
    if language:
        kwargs["language"] = language
    return whisperx.load_model(model_size, device, **kwargs)


@lru_cache(maxsize=4)
def _get_whisperx_align_model(language_code: str, device: str):
    _patch_torchaudio_compat()
    _patch_huggingface_hub_compat()
    _patch_torch_serialization_compat()
    import whisperx

    return whisperx.load_align_model(language_code=language_code, device=device)


def _transcribe_whisperx(
    audio_path: str | Path,
    model_size: str,
    language: str,
    batch_size: int,
    compute_type: str,
    device: str,
    enable_diarization: bool,
    hf_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    diarization_model: str,
) -> dict[str, Any]:
    import whisperx

    audio_path = Path(audio_path)
    model = _get_whisperx_model(model_size, device, compute_type, language or None)
    result = model.transcribe(str(audio_path), batch_size=batch_size, language=language or None)

    detected_language = result.get("language") or language or "ru"
    try:
        align_model, metadata = _get_whisperx_align_model(detected_language, device)
        result = whisperx.align(
            result.get("segments", []),
            align_model,
            metadata,
            str(audio_path),
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        result.setdefault("warnings", []).append(f"Не удалось выполнить word-level alignment: {exc}")

    diarization_enabled = False
    if enable_diarization:
        if not hf_token:
            result.setdefault("warnings", []).append(
                "Diarization пропущен: не задан HF_TOKEN. Укажите токен Hugging Face в .env или в интерфейсе."
            )
        else:
            try:
                diarization_turns = _run_pyannote_diarization(
                    audio_path=audio_path,
                    hf_token=hf_token,
                    device=device,
                    diarization_model=diarization_model,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                result = _assign_pyannote_speakers(result, diarization_turns)
                diarization_enabled = bool(diarization_turns)
                result["diarization_model"] = diarization_model
            except Exception as exc:
                result.setdefault("warnings", []).append(_friendly_diarization_error(exc, diarization_model))

    normalized = _normalize_segments(result.get("segments", []), probe_duration(audio_path), detected_language)
    normalized["language_probability"] = _float(result.get("language_probability", 0.0))
    normalized["asr_backend"] = "whisperx"
    normalized["diarization_enabled"] = diarization_enabled and any(
        u.get("speaker") != "SPEAKER_00" for u in normalized["utterances"]
    )
    normalized["warnings"] = result.get("warnings", [])
    return normalized


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    data = {}
    for name in ("text", "segments", "language", "duration"):
        if hasattr(obj, name):
            data[name] = getattr(obj, name)
    return data


def _transcribe_openai_diarize(
    audio_path: str | Path,
    language: str,
    openai_model: str,
    openai_api_key: str | None,
) -> dict[str, Any]:
    if not openai_api_key:
        raise RuntimeError("Для backend=openai-diarize нужен OPENAI_API_KEY в .env или в параметрах запуска.")

    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=openai_model,
            file=audio_file,
            language=language or None,
            response_format="diarized_json",
            chunking_strategy="auto",
        )
    data = _object_to_dict(response)
    segments = data.get("segments") or []
    normalized_segments: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            seg = _object_to_dict(seg)
        normalized_segments.append(
            {
                "text": seg.get("text", ""),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "speaker": seg.get("speaker") or seg.get("speaker_label") or "SPEAKER_00",
            }
        )
    if not normalized_segments and data.get("text"):
        normalized_segments.append({"text": data["text"], "start": 0.0, "end": probe_duration(audio_path), "speaker": "SPEAKER_00"})

    normalized = _normalize_segments(normalized_segments, data.get("duration") or probe_duration(audio_path), data.get("language") or language)
    normalized["asr_backend"] = "openai-diarize"
    normalized["diarization_enabled"] = any(u.get("speaker") != "SPEAKER_00" for u in normalized["utterances"])
    return normalized


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "small",
    language: str = "ru",
    backend: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    batch_size: int = 16,
    compute_type: str = "auto",
    device: str = "auto",
    enable_diarization: bool = True,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarization_model: str = "pyannote/speaker-diarization-community-1",
    openai_model: str = "gpt-4o-transcribe-diarize",
    openai_api_key: str | None = None,
) -> dict[str, Any]:
    """Transcribe audio with a modern pluggable ASR backend.

    backend:
    - auto: try WhisperX first, then fallback to faster-whisper;
    - whisperx: ASR + word alignment + optional speaker diarization;
    - faster-whisper: Windows-stable local ASR with optional pyannote diarization;
    - openai-diarize: API backend with built-in speaker labels.
    """
    audio_path = Path(audio_path)
    resolved_device = _resolve_device(device)
    resolved_compute_type = _resolve_compute_type(resolved_device, compute_type)
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if backend == "openai-diarize":
        return _transcribe_openai_diarize(audio_path, language, openai_model, openai_api_key)

    if backend in {"auto", "whisperx"}:
        try:
            return _transcribe_whisperx(
                audio_path=audio_path,
                model_size=model_size,
                language=language,
                batch_size=batch_size,
                compute_type=resolved_compute_type,
                device=resolved_device,
                enable_diarization=enable_diarization,
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                diarization_model=diarization_model,
            )
        except ImportError as exc:
            if backend == "whisperx":
                raise RuntimeError(
                    "WhisperX не установлен. Установите зависимости: pip install -r requirements_modern_asr.txt"
                ) from exc
        except Exception:
            if backend == "whisperx":
                raise

    return _transcribe_faster_whisper(
        audio_path=audio_path,
        model_size=model_size,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        compute_type=resolved_compute_type,
        device=resolved_device,
        enable_diarization=enable_diarization,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarization_model=diarization_model,
    )
