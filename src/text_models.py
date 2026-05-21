from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass
class ModelBundle:
    toxicity_pipe: object | None
    emotion_pipe: object | None
    device: int
    warnings: list[str]


_DEF_TOXICITY_MODEL = "cointegrated/rubert-tiny-toxicity"
_DEF_EMOTION_MODEL = "Djacon/rubert-tiny2-russian-emotion-detection"

_TOXIC_LABEL_PARTS = ("toxic", "insult", "obscene", "threat", "abusive", "hate", "aggression")
_NON_TOXIC_LABEL_PARTS = ("non", "not", "neutral", "normal", "clean", "нетокс")

_SUPPORT_MARKERS = ["молодец", "спокойно", "норм", "хорош", "держим", "давай", "получится", "красав", "спасибо"]
_COMMAND_MARKERS = ["идем", "иду", "беру", "отходи", "жми", "смок", "блок", "раш", "ульт", "слева", "справа", "прикрой", "пуш"]
_CRITICISM_MARKERS = ["зачем", "почему", "ошибка", "не надо", "плохо", "куда", "что ты", "опять"]

# Мягкие маркеры агрессии: могут быть эмоциональной критикой, но не всегда прямым оскорблением.
_AGGRESSION_MARKERS = [
    "дурак",
    "идиот",
    "заткнись",
    "бесишь",
    "туп",
    "слаб",
    "отвали",
    "мусор",
    "дебил",
    "урод",
    "клоун",
    "враги",
]

# Жёсткие маркеры ненормативной/оскорбительной лексики.
# Это не самостоятельная «модель», а слой калибровки поверх ML-модели: он исправляет очевидные
# ошибки вроде neutral для фраз с прямыми оскорблениями и матом.
_PROFANITY_STEMS = [
    "бляд",
    "блят",
    "сука",
    "сук",
    "мраз",
    "пидар",
    "пидарас",
    "пидор",
    "пидр",
    "уеб",
    "уёб",
    "уебк",
    "уёбк",
    "уеп",
    "уёп",
    "уйоп",
    "уйоб",
    "ебан",
    "ёбан",
    "ебат",
    "ёбат",
    "ебл",
    "ёбл",
    "нах",
    "хуй",
    "хуе",
    "хуё",
    "хер",
    "мудак",
    "мудил",
    "кончен",
]

_ANGER_MARKERS = [
    "бесит",
    "злюсь",
    "злость",
    "злой",
    "злая",
    "достал",
    "ненавиж",
    "раздраж",
    "капец",
    "ужас",
    "ярость",
] + _AGGRESSION_MARKERS + _PROFANITY_STEMS

_LATIN_PROFANITY_STEMS = ["fuck", "shit", "bitch", "idiot", "moron", "stupid"]


_DEFUSING_MARKERS = ["не ругай", "без мата", "давайте спокойно", "спокойно", "не токсич"]


def _get_device_arg() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


@lru_cache(maxsize=2)
def load_models(enable_emotion: bool = True, enable_toxicity: bool = True) -> ModelBundle:
    """Load text-classification models once per process.

    If a model cannot be loaded, the pipeline continues with rule-based fallback.
    This is useful for classroom demos without internet or GPU.
    """
    warnings: list[str] = []
    device = _get_device_arg()
    toxicity = None
    emotion = None
    try:
        from transformers import pipeline
    except Exception as exc:
        return ModelBundle(None, None, device, [f"Transformers недоступен, используются эвристики: {exc}"])

    if enable_toxicity:
        try:
            toxicity = pipeline(
                "text-classification",
                model=_DEF_TOXICITY_MODEL,
                tokenizer=_DEF_TOXICITY_MODEL,
                device=device,
            )
        except Exception as exc:
            warnings.append(f"Модель токсичности не загружена, используется эвристика: {exc}")

    if enable_emotion:
        try:
            emotion = pipeline(
                "text-classification",
                model=_DEF_EMOTION_MODEL,
                tokenizer=_DEF_EMOTION_MODEL,
                device=device,
            )
        except Exception as exc:
            warnings.append(f"Модель эмоций не загружена, используется эвристика: {exc}")

    return ModelBundle(toxicity_pipe=toxicity, emotion_pipe=emotion, device=device, warnings=warnings)


def _normalize_pipeline_output(raw: Any, expected_len: int) -> list[list[dict[str, Any]]]:
    if raw is None:
        return [[] for _ in range(expected_len)]
    if expected_len == 1 and isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return [raw]
    if isinstance(raw, list) and len(raw) == expected_len and all(isinstance(x, list) for x in raw):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return raw
    if isinstance(raw, list):
        return [raw] + [[] for _ in range(max(0, expected_len - 1))]
    return [[] for _ in range(expected_len)]


def _predict(pipe: object | None, texts: list[str], batch_size: int = 16) -> list[list[dict[str, Any]]]:
    if pipe is None or not texts:
        return [[] for _ in texts]
    try:
        raw = pipe(texts, top_k=None, truncation=True, batch_size=batch_size)
    except TypeError:
        raw = pipe(texts, top_k=None)
    except Exception:
        return [[] for _ in texts]
    return _normalize_pipeline_output(raw, len(texts))


def _label_text(label: str) -> str:
    return str(label or "").lower().replace("_", "-")


def _normalize_text(text: str) -> str:
    lowered = str(text or "").lower().replace("ё", "е")
    # Оставляем буквы/цифры/пробелы и !?, чтобы учитывать эмоциональную пунктуацию.
    return re.sub(r"[^a-zа-я0-9!?\s]+", " ", lowered)


def _count_stem_hits(text: str, stems: list[str]) -> int:
    normalized = _normalize_text(text)
    return sum(1 for stem in stems if stem.replace("ё", "е") in normalized)


def _has_defusing_context(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(marker in normalized for marker in _DEFUSING_MARKERS)


def _lexical_toxicity_score(text: str) -> float:
    """Conservative rule-based calibration for obvious abusive/profane utterances.

    Neural classifiers sometimes return neutral for noisy ASR output, slang, gaming speech or misspellings.
    This layer only raises the score for explicit profanity/aggression and does not reduce model scores.
    """
    if not text:
        return 0.0

    hard_hits = _count_stem_hits(text, _PROFANITY_STEMS) + _count_stem_hits(text, _LATIN_PROFANITY_STEMS)
    soft_hits = _count_stem_hits(text, _AGGRESSION_MARKERS)
    exclamations = min(2, str(text).count("!"))
    question_exclam = 1 if ("?!" in text or "!?" in text) else 0

    if hard_hits == 0 and soft_hits == 0:
        return 0.05

    score = 0.20 + 0.28 * hard_hits + 0.18 * soft_hits + 0.04 * exclamations + 0.05 * question_exclam
    if hard_hits >= 2:
        score = max(score, 0.90)
    elif hard_hits == 1:
        score = max(score, 0.78)
    elif soft_hits >= 2:
        score = max(score, 0.70)

    if _has_defusing_context(text):
        score *= 0.65

    return round(min(0.995, max(0.0, score)), 4)


def _extract_toxicity(output: list[dict[str, Any]], text: str) -> tuple[str, float, str]:
    candidates: list[tuple[str, float]] = []
    for item in output or []:
        label = _label_text(item.get("label", ""))
        score = float(item.get("score", 0.0) or 0.0)
        if any(part in label for part in _NON_TOXIC_LABEL_PARTS):
            continue
        if any(part in label for part in _TOXIC_LABEL_PARTS):
            candidates.append((str(item.get("label", "toxic")), score))

    lexical_score = _lexical_toxicity_score(text)

    if candidates:
        label, model_score = max(candidates, key=lambda x: x[1])
        score = max(float(model_score), lexical_score)
        source = "model+lexicon" if lexical_score > model_score + 0.05 else "model"
        if lexical_score >= 0.70 and label.lower() in {"neutral", "non-toxic", "not-toxic"}:
            label = "lexicon_aggression"
        return label, round(score, 4), source

    if output:
        best = max(output, key=lambda x: float(x.get("score", 0.0) or 0.0))
        label = str(best.get("label", "unknown"))
        label_l = _label_text(label)
        if any(part in label_l for part in _NON_TOXIC_LABEL_PARTS):
            inverted = round(1.0 - float(best.get("score", 0.0) or 0.0), 4)
            score = max(inverted, lexical_score)
            source = "model_inverted+lexicon" if lexical_score > inverted + 0.05 else "model_inverted"
            label = "lexicon_aggression" if lexical_score >= 0.70 else label
            return label, round(score, 4), source

    score = _rule_based_toxicity(text)
    return "rule_based_toxicity", score, "rules"


def _rule_based_toxicity(text: str) -> float:
    return _lexical_toxicity_score(text)


def _extract_emotion(output: list[dict[str, Any]], text: str, toxicity_score: float | None = None) -> tuple[str, float, str]:
    lexical_toxicity = _lexical_toxicity_score(text)
    toxic_score = max(float(toxicity_score or 0.0), lexical_toxicity)
    anger_hits = _count_stem_hits(text, _ANGER_MARKERS) + _count_stem_hits(text, _LATIN_PROFANITY_STEMS)

    # Калибровка: прямая брань/оскорбления в командной речи почти всегда ближе к anger/frustration,
    # даже если текстовая модель эмоций из-за шума ASR выдала neutral/joy.
    if toxic_score >= 0.70 and anger_hits > 0:
        return "anger", round(max(0.75, min(0.98, toxic_score)), 4), "rules_override"

    if output:
        best = max(output, key=lambda x: float(x.get("score", 0.0) or 0.0))
        label = str(best.get("label", "unknown"))
        score = round(float(best.get("score", 0.0) or 0.0), 4)
        if _label_text(label) in {"neutral", "joy"} and lexical_toxicity >= 0.55:
            return "anger", round(max(score, 0.70), 4), "model_overridden_by_lexicon"
        return label, score, "model"

    lowered = _normalize_text(text)
    if any(marker.replace("ё", "е") in lowered for marker in _ANGER_MARKERS):
        return "anger", 0.65, "rules"
    if any(marker in lowered for marker in _SUPPORT_MARKERS):
        return "joy", 0.55, "rules"
    if "?" in text:
        return "neutral_question", 0.45, "rules"
    return "neutral", 0.5, "rules"


def enrich_utterances(utterances: list[dict[str, Any]], models: ModelBundle, batch_size: int = 16) -> list[dict[str, Any]]:
    texts = [str(item.get("text", "")) for item in utterances]
    tox_outputs = _predict(models.toxicity_pipe, texts, batch_size=batch_size)
    emo_outputs = _predict(models.emotion_pipe, texts, batch_size=batch_size)

    enriched: list[dict[str, Any]] = []
    for item, tox_raw, emo_raw in zip(utterances, tox_outputs, emo_outputs):
        text = str(item.get("text", ""))
        toxicity_label, toxicity_score, toxicity_source = _extract_toxicity(tox_raw, text)
        emotion_label, emotion_score, emotion_source = _extract_emotion(emo_raw, text, toxicity_score)

        enriched_item = dict(item)
        enriched_item["toxicity_label"] = toxicity_label
        enriched_item["toxicity_score"] = round(float(toxicity_score), 4)
        enriched_item["toxicity_source"] = toxicity_source
        enriched_item["emotion_label"] = emotion_label
        enriched_item["emotion_score"] = round(float(emotion_score), 4)
        enriched_item["emotion_source"] = emotion_source
        enriched_item["utterance_type"] = classify_utterance_type(text, float(toxicity_score))
        enriched_item["communication_function"] = classify_communication_function(text, float(toxicity_score))
        enriched.append(enriched_item)

    return enriched


def classify_utterance_type(text: str, toxicity_score: float) -> str:
    lowered = _normalize_text(text)
    lexical_score = _lexical_toxicity_score(text)
    if toxicity_score >= 0.65 or lexical_score >= 0.65 or any(x in lowered for x in _AGGRESSION_MARKERS + _PROFANITY_STEMS):
        return "conflict_or_aggression"
    if any(x in lowered for x in _SUPPORT_MARKERS):
        return "support"
    if any(x in lowered for x in _COMMAND_MARKERS):
        return "coordination_command"
    if "?" in text:
        return "question"
    if any(x in lowered for x in _CRITICISM_MARKERS):
        return "criticism_or_problem"
    return "information_or_comment"


def classify_communication_function(text: str, toxicity_score: float) -> str:
    utterance_type = classify_utterance_type(text, toxicity_score)
    mapping = {
        "conflict_or_aggression": "деструктивное взаимодействие",
        "support": "поддержка команды",
        "coordination_command": "координация действий",
        "question": "уточнение информации",
        "criticism_or_problem": "критика/обсуждение ошибки",
        "information_or_comment": "информационная реплика",
    }
    return mapping.get(utterance_type, "прочее")
