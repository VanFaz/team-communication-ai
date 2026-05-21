from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

from .audio_utils import ensure_dir, format_ts
from .profanity import mask_profanity


HTML_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчёт по анализу командной коммуникации</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #111827; background: #ffffff; }
    h1, h2, h3 { color: #111827; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
    .card { background: #f9fafb; padding: 16px; border: 1px solid #e5e7eb; border-radius: 14px; margin-bottom: 16px; }
    .metric { font-size: 28px; font-weight: 700; }
    .label { color: #4b5563; font-size: 13px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border: 1px solid #d1d5db; padding: 8px; font-size: 14px; vertical-align: top; }
    th { background: #e5e7eb; text-align: left; }
    tr.risk { background: #fff7ed; }
    .small { color: #4b5563; font-size: 13px; }
    .pill { display: inline-block; border-radius: 999px; padding: 2px 8px; background: #e5e7eb; font-size: 12px; }
    .bar { height: 10px; border-radius: 999px; background: #e5e7eb; overflow: hidden; }
    .bar span { display: block; height: 100%; background: #9ca3af; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <h1>Отчёт по анализу командной коммуникации</h1>
  <div class="card">
    <p><strong>Файл:</strong> {{ meta.input_file }}</p>
    <p><strong>ASR backend:</strong> <code>{{ meta.asr_backend }}</code>; <strong>Diarization:</strong> {{ 'включён' if meta.diarization_enabled else 'не выполнен' }}</p>
    <p><strong>Язык:</strong> {{ meta.language }}; <strong>Продолжительность:</strong> {{ meta.duration_sec }} сек.</p>
    <p><strong>Интерпретация:</strong> {{ interpretation }}</p>
    {% if meta.warnings %}
      <p class="small"><strong>Предупреждения:</strong> {{ meta.warnings | join('; ') }}</p>
    {% endif %}
  </div>

  <div class="grid">
    <div class="card"><div class="label">Индекс качества коммуникации</div><div class="metric">{{ summary.communication_quality_score }}/100</div></div>
    <div class="card"><div class="label">Количество реплик</div><div class="metric">{{ summary.num_utterances }}</div></div>
    <div class="card"><div class="label">Средняя токсичность</div><div class="metric">{{ summary.avg_toxicity }}</div></div>
    <div class="card"><div class="label">Говорящих</div><div class="metric">{{ summary.speaker_metrics.num_speakers if summary.speaker_metrics else 0 }}</div></div>
  </div>

  <div class="card">
    <h2>Ключевые метрики</h2>
    <ul>
      <li>Доминирующая эмоция: <strong>{{ summary.dominant_emotion }}</strong></li>
      <li>Доминирующий тип реплик: <strong>{{ summary.dominant_type }}</strong></li>
      <li>Темп речи: <strong>{{ summary.speech_rate_utterances_per_min }}</strong> реплик/мин</li>
      <li>Доля поддержки: <strong>{{ summary.support_rate }}</strong></li>
      <li>Доля конфликтных реплик: <strong>{{ summary.conflict_rate }}</strong></li>
      <li>Доля вопросов: <strong>{{ summary.question_rate }}</strong></li>
    </ul>
  </div>

  <div class="card">
    <h2>Флаги риска</h2>
    {% if summary.risk_flags %}
      <ul>{% for flag in summary.risk_flags %}<li>{{ flag }}</li>{% endfor %}</ul>
    {% else %}
      <p>Не обнаружены.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Метрики по говорящим</h2>
    {% if speaker_rows %}
      <table>
        <thead><tr><th>Говорящий</th><th>Реплики</th><th>Время речи, сек.</th><th>Доля реплик</th></tr></thead>
        <tbody>
        {% for speaker in speaker_rows %}
          <tr><td>{{ speaker.speaker }}</td><td>{{ speaker.turns }}</td><td>{{ speaker.speaking_time_sec }}</td><td><div class="bar"><span style="width: {{ speaker.turn_share_pct }}%"></span></div> {{ speaker.turn_share_pct }}%</td></tr>
        {% endfor %}
        </tbody>
      </table>
      <p class="small">Возможные перебивания/пересечения речи: {{ summary.speaker_metrics.interruptions_count }}; время пересечений: {{ summary.speaker_metrics.overlap_time_sec }} сек.</p>
    {% else %}
      <p>Нет данных по говорящим.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Фрагменты с наибольшим риском</h2>
    {% if risky_rows %}
      <table>
        <thead><tr><th>Время</th><th>Говорящий</th><th>Токсичность</th><th>Источник токс.</th><th>Эмоция</th><th>Источник эмоц.</th><th>Текст</th></tr></thead>
        <tbody>
        {% for row in risky_rows %}
          <tr class="risk"><td>{{ row.start_fmt }} - {{ row.end_fmt }}</td><td>{{ row.speaker }}</td><td>{{ row.toxicity_score }}</td><td>{{ row.toxicity_source }}</td><td>{{ row.emotion_label }}</td><td>{{ row.emotion_source }}</td><td>{{ row.text }}</td></tr>
        {% endfor %}
        </tbody>
      </table>
    {% else %}
      <p>Рискованные фрагменты не выделены.</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Таблица реплик</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Время</th><th>Говорящий</th><th>Текст</th><th>Функция</th><th>Тип</th><th>Токсичность</th><th>Источник токс.</th><th>Эмоция</th><th>Источник эмоц.</th>
        </tr>
      </thead>
      <tbody>
      {% for row in rows %}
        <tr class="{{ 'risk' if row.toxicity_score|float >= 0.5 or row.utterance_type == 'conflict_or_aggression' else '' }}">
          <td>{{ row.id }}</td>
          <td>{{ row.start_fmt }} - {{ row.end_fmt }}</td>
          <td><span class="pill">{{ row.speaker }}</span></td>
          <td>{{ row.text }}</td>
          <td>{{ row.communication_function }}</td>
          <td>{{ row.utterance_type }}</td>
          <td>{{ row.toxicity_score }}</td>
          <td>{{ row.toxicity_source }}</td>
          <td>{{ row.emotion_label }}</td>
          <td>{{ row.emotion_source }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def _scalar_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("words", None)
    return result


def _prepare_rows(utterances: list[dict]) -> list[dict]:
    rows = []
    for u in utterances:
        row = _scalar_row(u)
        row["text"] = mask_profanity(row.get("text", ""))
        row["start_fmt"] = format_ts(u.get("start"))
        row["end_fmt"] = format_ts(u.get("end"))
        rows.append(row)
    return rows


def _speaker_rows(summary: dict) -> list[dict[str, Any]]:
    speaker_metrics = summary.get("speaker_metrics") or {}
    speakers = speaker_metrics.get("speakers") or {}
    total_turns = max(1, sum(int(v.get("turns", 0)) for v in speakers.values()))
    rows = []
    for speaker, data in speakers.items():
        turns = int(data.get("turns", 0))
        rows.append(
            {
                "speaker": speaker,
                "turns": turns,
                "speaking_time_sec": round(float(data.get("speaking_time_sec", 0.0)), 2),
                "turn_share_pct": round(turns / total_turns * 100, 1),
            }
        )
    return sorted(rows, key=lambda x: (-x["turns"], x["speaker"]))


def _markdown_summary(result: dict) -> str:
    summary = result["summary"]
    lines = [
        "# Краткое резюме анализа командной коммуникации",
        "",
        f"Файл: {result['meta']['input_file']}",
        f"ASR backend: {result['meta']['asr_backend']}",
        f"Diarization: {'включён' if result['meta'].get('diarization_enabled') else 'не выполнен'}",
        f"Индекс качества коммуникации: {summary.get('communication_quality_score')}/100",
        f"Средняя токсичность: {summary.get('avg_toxicity')}",
        f"Доминирующий тип реплик: {summary.get('dominant_type')}",
        "",
        "## Интерпретация",
        result.get("interpretation", ""),
        "",
        "## Флаги риска",
    ]
    flags = summary.get("risk_flags") or []
    lines.extend([f"- {flag}" for flag in flags] or ["- Не обнаружены"])
    return "\n".join(lines) + "\n"


def save_outputs(result: dict, output_dir: str | Path) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    rows = _prepare_rows(result["utterances"])
    risky_rows = sorted(rows, key=lambda r: (float(r.get("toxicity_score", 0.0)), r.get("utterance_type") == "conflict_or_aggression"), reverse=True)[:10]
    speaker_rows = _speaker_rows(result["summary"])

    csv_path = output_dir / "utterances.csv"
    json_path = output_dir / "analysis.json"
    html_path = output_dir / "report.html"
    md_path = output_dir / "summary.md"

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    template = Template(HTML_TEMPLATE)
    html = template.render(
        meta=result["meta"],
        summary=result["summary"],
        interpretation=result["interpretation"],
        rows=rows,
        risky_rows=risky_rows,
        speaker_rows=speaker_rows,
    )
    html_path.write_text(html, encoding="utf-8")
    md_path.write_text(_markdown_summary(result), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "html": str(html_path),
        "summary_md": str(md_path),
    }
