"""
gradio_app.py — Medical Triage Demo
====================================
Run locally:
    pip install gradio transformers torch joblib
    python gradio_app.py

Deploy to HuggingFace Spaces:
    Create a Gradio Space and upload:
        gradio_app.py, requirements.txt, checkpoints/
"""
from __future__ import annotations

import re
import joblib
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME      = "DeepPavlov/rubert-base-cased"
CHECKPOINT_PATH = Path("checkpoints/model_b_best.pt")
ENCODER_PATH    = Path("checkpoints/label_encoder.joblib")
MAX_LENGTH      = 128
CONF_THRESHOLD  = 0.70

HIGH_ACUITY = {"кардиология", "онкология", "неврология", "хирургия", "травматология"}

# ── Load artefacts (once at startup) ─────────────────────────────────────────
for p in (CHECKPOINT_PATH, ENCODER_PATH):
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.\n"
            "Run medical_complaint_classification.ipynb first to generate checkpoints."
        )

le        = joblib.load(ENCODER_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
    ignore_mismatched_sizes=True,
)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
)
model.eval()


# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Replicate training-time preprocessing."""
    text = str(text).lower()
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_full(complaint: str, top_k: int = 5) -> tuple[str, list]:
    """
    Run inference and return (markdown_output, bar_chart_data).

    Returns
    -------
    md_output   : Formatted markdown string for the main output box.
    chart_data  : List of [specialty, probability] for the bar chart component.
    """
    if not complaint.strip():
        return "⬆️ Введите жалобу пациента выше.", []

    cleaned = clean_text(complaint)
    enc = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        proba = torch.softmax(model(**enc).logits, dim=-1).squeeze().numpy()

    top_idx   = np.argsort(proba)[::-1][:top_k]
    top_preds = [(le.classes_[i], round(float(proba[i]), 4)) for i in top_idx]

    top_spec  = top_preds[0][0]
    top_conf  = top_preds[0][1]
    confident = top_conf >= CONF_THRESHOLD
    is_urgent = top_spec in HIGH_ACUITY

    # ── Markdown output ───────────────────────────────────────────────────────
    urgency_icon = "🚨" if is_urgent else "🏥"
    conf_icon    = "✅" if confident else "⚠️"

    lines = [
        f"## {urgency_icon} {top_spec.capitalize()}",
        "",
        f"**Уверенность:** {top_conf:.0%}  {conf_icon}",
        "",
    ]

    if confident:
        lines.append(f"**Рекомендация:** направить к специалисту — **{top_spec}**")
    else:
        lines.append(
            f"**Рекомендация:** низкая уверенность ({top_conf:.0%}) — "
            f"направить к **терапевту** для ручной триажировки"
        )

    if is_urgent and confident:
        lines += ["", "---", "🚨 **Высокоприоритетная специальность** — не откладывать запись"]

    lines += ["", "---", "**Топ предсказаний:**", ""]
    for rank, (spec, prob) in enumerate(top_preds, 1):
        bar = "█" * int(prob * 25)
        lines.append(f"{rank}. `{spec:<28}` {prob:.1%}  {bar}")

    chart_data = [[spec, round(prob * 100, 1)] for spec, prob in top_preds]

    return "\n".join(lines), chart_data


def triage_interface(complaint: str) -> tuple[str, dict]:
    md_out, chart_data = predict_full(complaint)
    bar_chart = gr.BarPlot.update(
        value={"specialty": [r[0] for r in chart_data],
               "probability": [r[1] for r in chart_data]},
        x="specialty",
        y="probability",
        title="Вероятности по специальностям (%)",
        y_lim=[0, 100],
        color="steelblue",
    ) if chart_data else gr.BarPlot.update()
    return md_out, bar_chart


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Medical Triage RU", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏥 Медицинский триаж — Маршрутизация по жалобе\n\n"
        "Введите жалобу пациента на **русском языке** — система определит специальность врача.\n\n"
        f"> Модель: `{MODEL_NAME}` | Специальностей: {len(le.classes_)} | "
        f"Порог уверенности: {CONF_THRESHOLD:.0%}"
    )

    with gr.Row():
        with gr.Column(scale=1):
            complaint_input = gr.Textbox(
                label="Жалоба пациента",
                placeholder="Опишите симптомы подробно...",
                lines=4,
                max_lines=8,
            )
            submit_btn = gr.Button("🔍 Определить специальность", variant="primary")

            gr.Markdown("**Примеры жалоб:**")
            gr.Examples(
                examples=[
                    ["У меня сильно болит сердце, давление 160 на 100, не могу дышать"],
                    ["Заметил опухоль на коже — тёмная с неровными краями, растёт два месяца"],
                    ["Постоянные головные боли, головокружение, теряю равновесие при ходьбе"],
                    ["Хочу похудеть, не знаю с чего начать, какая диета лучше"],
                    ["Ребёнку 3 года, температура 39, кашель третий день, насморк"],
                    ["Плохо сплю, тревожные мысли, не могу расслабиться уже месяц"],
                    ["Зуб болит третий день, опухла щека, больно открывать рот"],
                ],
                inputs=complaint_input,
                label="",
            )

        with gr.Column(scale=1):
            result_output = gr.Markdown(
                value="Введите жалобу и нажмите кнопку.",
                label="Результат",
            )
            bar_chart = gr.BarPlot(
                x="specialty",
                y="probability",
                title="Вероятности (%)",
                visible=True,
                height=250,
            )

    submit_btn.click(
        fn=triage_interface,
        inputs=complaint_input,
        outputs=[result_output, bar_chart],
    )
    complaint_input.submit(
        fn=triage_interface,
        inputs=complaint_input,
        outputs=[result_output, bar_chart],
    )

    gr.Markdown(
        "---\n"
        "⚕️ *Инструмент для исследовательских целей. "
        "Не использовать для принятия клинических решений без верификации врачом.*"
    )

if __name__ == "__main__":
    demo.launch(share=False)
