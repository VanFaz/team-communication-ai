from __future__ import annotations

import json
import os
from typing import Any

import gradio as gr
import pandas as pd

from src.pipeline import run_pipeline
from src.profanity import mask_profanity


def _none_if_empty(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _input_path(file_obj: Any) -> str:
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    raise gr.Error("Не удалось прочитать путь к файлу.")


def analyze(
    file_obj,
    backend: str,
    model_size: str,
    language: str,
    enable_diarization: bool,
    min_speakers: float | None,
    max_speakers: float | None,
    diarization_model: str,
    enable_emotion: bool,
    device: str,
    compute_type: str,
    batch_size: float,
    hf_token: str,
    openai_api_key: str,
):
    if file_obj is None:
        raise gr.Error("Загрузите аудио или видео файл.")

    result = run_pipeline(
        input_path=_input_path(file_obj),
        work_dir="output",
        backend=backend,
        model_size=model_size,
        language=language.strip() or "ru",
        enable_diarization=enable_diarization,
        min_speakers=int(min_speakers) if min_speakers else None,
        max_speakers=int(max_speakers) if max_speakers else None,
        diarization_model=diarization_model.strip() or "pyannote/speaker-diarization-3.1",
        enable_emotion=enable_emotion,
        device=device,
        compute_type=compute_type,
        batch_size=int(batch_size or 16),
        hf_token=_none_if_empty(hf_token) or os.getenv("HF_TOKEN"),
        openai_api_key=_none_if_empty(openai_api_key) or os.getenv("OPENAI_API_KEY"),
    )

    table_rows = []
    for row in result["utterances"]:
        table_row = {k: v for k, v in row.items() if k != "words"}
        table_row["text"] = mask_profanity(table_row.get("text", ""))
        table_rows.append(table_row)
    df = pd.DataFrame(table_rows)
    summary_text = json.dumps(
        {
            "summary": result["summary"],
            "interpretation": result["interpretation"],
            "artifacts": result["artifacts"],
            "warnings": result["meta"].get("warnings", []),
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        summary_text,
        df,
        result["artifacts"]["html"],
        result["artifacts"]["csv"],
        result["artifacts"]["json"],
        result["artifacts"]["summary_md"],
    )


with gr.Blocks(title="ИИ-анализ командной коммуникации") as demo:
    gr.Markdown(
        "# ИИ-анализ командной коммуникации\n"
        "Современный прототип для курсового проекта: загрузка записи → ASR → word-level timestamps → diarization по говорящим → NLP-анализ реплик → отчёт."
    )
    with gr.Row():
        file_input = gr.File(label="Аудио или видео файл")
        with gr.Column():
            backend = gr.Dropdown(
                ["auto", "whisperx", "faster-whisper", "openai-diarize"],
                value="faster-whisper",
                label="ASR backend",
                info="Windows-режим: faster-whisper + pyannote. WhisperX только при отдельной установке modern-зависимостей.",
            )
            model_size = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"], value="small", label="Локальная модель")
            language = gr.Textbox(value="ru", label="Код языка")
            enable_diarization = gr.Checkbox(value=True, label="Разделять говорящих")
            enable_emotion = gr.Checkbox(value=True, label="Оценивать эмоции по тексту")
            run_btn = gr.Button("Запустить анализ", variant="primary")

    with gr.Accordion("Расширенные настройки", open=False):
        with gr.Row():
            min_speakers = gr.Number(value=None, label="Мин. говорящих", precision=0)
            max_speakers = gr.Number(value=None, label="Макс. говорящих", precision=0)
            batch_size = gr.Number(value=16, label="Batch size", precision=0)
        diarization_model = gr.Textbox(value="pyannote/speaker-diarization-3.1", label="pyannote diarization model")
        with gr.Row():
            device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
            compute_type = gr.Dropdown(["auto", "int8", "float16", "float32"], value="auto", label="Compute type")
        with gr.Row():
            hf_token = gr.Textbox(value="", label="HF_TOKEN для pyannote/WhisperX", type="password")
            openai_api_key = gr.Textbox(value="", label="OPENAI_API_KEY для openai-diarize", type="password")

    summary = gr.Code(label="Краткий результат", language="json")
    table = gr.Dataframe(label="Таблица реплик")
    with gr.Row():
        html_file = gr.File(label="HTML-отчёт")
        csv_file = gr.File(label="CSV-таблица")
        json_file = gr.File(label="JSON-результат")
        md_file = gr.File(label="Markdown-резюме")

    run_btn.click(
        analyze,
        inputs=[
            file_input,
            backend,
            model_size,
            language,
            enable_diarization,
            min_speakers,
            max_speakers,
            diarization_model,
            enable_emotion,
            device,
            compute_type,
            batch_size,
            hf_token,
            openai_api_key,
        ],
        outputs=[summary, table, html_file, csv_file, json_file, md_file],
    )


if __name__ == "__main__":
    demo.launch()
