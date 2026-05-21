from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _ratio(part: float, total: float) -> float:
    return round(part / total, 4) if total else 0.0


def compute_speaker_metrics(utterances: list[dict]) -> dict:
    by_speaker: dict[str, dict[str, float | int]] = defaultdict(lambda: {"turns": 0, "speaking_time_sec": 0.0})
    sorted_items = sorted(utterances, key=lambda x: (_num(x.get("start")), _num(x.get("end"))))

    interruptions = 0
    overlap_time = 0.0
    interruptions_by_speaker: Counter[str] = Counter()

    previous = None
    for item in sorted_items:
        speaker = str(item.get("speaker") or "SPEAKER_00")
        start = _num(item.get("start"))
        end = _num(item.get("end"), start)
        duration = max(0.0, end - start)
        by_speaker[speaker]["turns"] = int(by_speaker[speaker]["turns"]) + 1
        by_speaker[speaker]["speaking_time_sec"] = round(float(by_speaker[speaker]["speaking_time_sec"]) + duration, 3)

        if previous is not None:
            prev_speaker = str(previous.get("speaker") or "SPEAKER_00")
            prev_end = _num(previous.get("end"))
            if speaker != prev_speaker and start < prev_end - 0.2:
                interruptions += 1
                overlap_time += max(0.0, prev_end - start)
                interruptions_by_speaker[speaker] += 1
        previous = item

    turn_counts = [int(v["turns"]) for v in by_speaker.values()]
    time_values = [float(v["speaking_time_sec"]) for v in by_speaker.values()]
    speaker_balance_turns = round(min(turn_counts) / max(turn_counts), 4) if turn_counts and max(turn_counts) else 1.0
    speaker_balance_time = round(min(time_values) / max(time_values), 4) if time_values and max(time_values) else 1.0

    return {
        "num_speakers": len(by_speaker),
        "speakers": dict(sorted(by_speaker.items())),
        "speaker_balance_turns": speaker_balance_turns,
        "speaker_balance_time": speaker_balance_time,
        "interruptions_count": interruptions,
        "overlap_time_sec": round(overlap_time, 3),
        "interruptions_by_speaker": dict(interruptions_by_speaker),
    }


def _quality_score(avg_toxicity: float, support_rate: float, conflict_rate: float, question_rate: float, speaker_balance: float, interruptions_rate: float) -> int:
    score = 100.0
    score -= min(45.0, avg_toxicity * 55.0)
    score -= min(30.0, conflict_rate * 120.0)
    score -= min(12.0, interruptions_rate * 60.0)
    score -= max(0.0, 0.6 - speaker_balance) * 20.0
    score += min(8.0, support_rate * 40.0)
    score += min(5.0, question_rate * 15.0)
    return int(max(0, min(100, round(score))))


def compute_summary(utterances: list[dict], duration: float) -> dict:
    duration = _num(duration)
    if not utterances:
        return {
            "num_utterances": 0,
            "duration_sec": round(duration, 2),
            "avg_toxicity": 0.0,
            "dominant_emotion": "unknown",
            "dominant_type": "unknown",
            "communication_quality_score": 0,
            "speech_rate_utterances_per_min": 0.0,
            "risk_flags": ["Нет распознанных реплик"],
        }

    n = len(utterances)
    avg_toxicity = mean(_num(u.get("toxicity_score")) for u in utterances)
    emotion_counts = Counter(str(u.get("emotion_label") or "unknown") for u in utterances)
    type_counts = Counter(str(u.get("utterance_type") or "unknown") for u in utterances)
    speaker_metrics = compute_speaker_metrics(utterances)

    support_count = type_counts.get("support", 0)
    conflict_count = type_counts.get("conflict_or_aggression", 0)
    question_count = type_counts.get("question", 0)
    support_rate = _ratio(support_count, n)
    conflict_rate = _ratio(conflict_count, n)
    question_rate = _ratio(question_count, n)
    interruptions_rate = _ratio(speaker_metrics.get("interruptions_count", 0), n)
    speaker_balance = min(
        float(speaker_metrics.get("speaker_balance_turns", 1.0)),
        float(speaker_metrics.get("speaker_balance_time", 1.0)),
    )

    risk_flags: list[str] = []
    if avg_toxicity >= 0.55:
        risk_flags.append("Высокий средний уровень токсичности")
    elif avg_toxicity >= 0.3:
        risk_flags.append("Умеренный уровень токсичности")
    if conflict_count >= max(2, n * 0.15):
        risk_flags.append("Часто встречаются конфликтные или агрессивные реплики")
    if emotion_counts.get("anger", 0) >= max(2, n * 0.15):
        risk_flags.append("Часто встречается гнев")
    if support_count == 0 and n >= 8:
        risk_flags.append("Не обнаружены поддерживающие реплики")
    if speaker_metrics.get("num_speakers", 1) > 1 and speaker_balance < 0.35:
        risk_flags.append("Наблюдается дисбаланс участия между говорящими")
    if speaker_metrics.get("interruptions_count", 0) >= max(2, n * 0.12):
        risk_flags.append("Обнаружены частые пересечения речи/возможные перебивания")

    quality_score = _quality_score(avg_toxicity, support_rate, conflict_rate, question_rate, speaker_balance, interruptions_rate)

    return {
        "num_utterances": n,
        "duration_sec": round(duration, 2),
        "avg_toxicity": round(avg_toxicity, 4),
        "dominant_emotion": emotion_counts.most_common(1)[0][0],
        "dominant_type": type_counts.most_common(1)[0][0],
        "communication_quality_score": quality_score,
        "speech_rate_utterances_per_min": round(n / max(duration / 60.0, 1e-6), 2),
        "support_rate": support_rate,
        "conflict_rate": conflict_rate,
        "question_rate": question_rate,
        "type_distribution": dict(type_counts),
        "emotion_distribution": dict(emotion_counts),
        "speaker_metrics": speaker_metrics,
        "risk_flags": risk_flags,
    }


def build_interpretation(summary: dict) -> str:
    flags = summary.get("risk_flags", [])
    avg_toxicity = float(summary.get("avg_toxicity", 0.0) or 0.0)
    score = int(summary.get("communication_quality_score", 0) or 0)
    dominant_type = summary.get("dominant_type", "unknown")

    if score >= 80 and avg_toxicity < 0.25:
        base = "Коммуникация выглядит преимущественно конструктивной: преобладают рабочие реплики, уровень токсичности низкий."
    elif score >= 60:
        base = "Коммуникация в целом рабочая, но присутствуют отдельные маркеры напряжения или дисбаланса."
    elif score >= 40:
        base = "Коммуникация содержит заметные признаки напряжения; требуется ручной просмотр фрагментов с повышенным риском."
    else:
        base = "Коммуникация выглядит проблемной по формальным маркерам: обнаружены признаки деструктивных взаимодействий."

    if dominant_type == "coordination_command":
        base += " Основная функция реплик — координация действий."
    elif dominant_type == "support":
        base += " В записи заметна поддерживающая коммуникация."
    elif dominant_type == "conflict_or_aggression":
        base += " Наиболее частый тип реплик связан с конфликтом или агрессией."

    if not flags:
        return base + " Явных флагов риска не обнаружено."
    return base + " Флаги риска: " + "; ".join(flags) + "."
