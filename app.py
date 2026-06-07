"""
app.py — FastAPI Production Endpoint
=====================================
Назначение: REST API для продакшн-систем (EHR, CRM, очередь триажа).
Это НЕ интерактивное демо — для демо используйте gradio_app.py.

Запуск:
    pip install fastapi uvicorn pydantic>=2.7.0 transformers torch joblib
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

Пример запроса:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "болит сердце, давление 160 на 100"}'

Docker:
    docker build -t medical-triage .
    docker run -p 8000:8000 medical-triage
"""
from __future__ import annotations

import re
import time
import logging
import hashlib
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME      = "DeepPavlov/rubert-base-cased"
CHECKPOINT_PATH = Path("checkpoints/model_b_best.pt")
ENCODER_PATH    = Path("checkpoints/label_encoder.joblib")
MAX_LENGTH      = 128
CONF_THRESHOLD  = 0.70   # below → route to терапевт for manual review
FALLBACK_CLASS  = "терапия"

HIGH_ACUITY = frozenset({
    "кардиология", "онкология", "неврология",
    "хирургия", "травматология",
})

DEVICE = torch.device("cpu")   # CPU is fine for inference; add .cuda() for GPU

# ── Load artefacts at startup (not per-request) ───────────────────────────────
for p in (CHECKPOINT_PATH, ENCODER_PATH):
    if not p.exists():
        raise RuntimeError(
            f"Required file '{p}' not found.\n"
            "Run medical_complaint_classification.ipynb to generate checkpoints."
        )

le        = joblib.load(ENCODER_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
    ignore_mismatched_sizes=True,
)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()
logger.info("Model loaded: %s | Classes: %d", MODEL_NAME, len(le.classes_))


# ── Schemas ───────────────────────────────────────────────────────────────────
class ComplaintRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Жалоба пациента на русском языке",
        examples=["болит сердце, давление 160 на 100"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=23,
        description="Количество специальностей в ответе",
    )


class SpecialtyScore(BaseModel):
    specialty: str
    probability: float


class PredictionResponse(BaseModel):
    # Primary output
    specialty:     str     = Field(description="Рекомендуемая специальность")
    confidence:    float   = Field(description="Уверенность модели (0–1)")
    is_confident:  bool    = Field(description="True если confidence ≥ порога")
    is_high_acuity: bool   = Field(description="True для высокоприоритетных специальностей")

    # Routing advice
    recommendation: str    = Field(description="Человекочитаемая рекомендация")

    # Top-k predictions
    top_k: List[SpecialtyScore]

    # Audit fields (never contains raw patient text)
    text_hash:      str    = Field(description="SHA-256 хеш текста жалобы")
    latency_ms:     float  = Field(description="Время инференса в мс")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Replicate training-time preprocessing exactly."""
    text = str(text).lower()
    text = re.sub(r"[^а-яёa-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical Triage API",
    description=(
        "Маршрутизация пациентов по специальности на основе текста жалобы. "
        "Модель: rubert-base-cased, 23 специальности, русский язык."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ComplaintRequest) -> PredictionResponse:
    """
    Предсказать медицинскую специальность по тексту жалобы пациента.

    - **text**: текст жалобы на русском языке (3–2000 символов)
    - **top_k**: сколько специальностей вернуть в ответе (по умолчанию 3)
    """
    t0 = time.perf_counter()

    # Audit hash — never log raw patient text
    text_hash = hashlib.sha256(request.text.encode()).hexdigest()[:16]

    cleaned = clean_text(request.text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Текст жалобы пуст после очистки.")

    enc = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits

    proba     = torch.softmax(logits, dim=-1).cpu().squeeze().numpy()
    top_idx   = np.argsort(proba)[::-1][: request.top_k]
    top_preds = [
        SpecialtyScore(
            specialty=le.classes_[i],
            probability=round(float(proba[i]), 4),
        )
        for i in top_idx
    ]

    top_spec  = top_preds[0].specialty
    top_conf  = top_preds[0].probability
    confident = top_conf >= CONF_THRESHOLD

    if confident:
        routed    = top_spec
        rec       = f"Направить к специалисту: {routed} (уверенность {top_conf:.0%})"
    else:
        routed    = FALLBACK_CLASS
        rec       = (
            f"Низкая уверенность ({top_conf:.0%}) — направить к терапевту "
            f"для ручной триажировки. Наиболее вероятно: {top_spec}"
        )

    latency = (time.perf_counter() - t0) * 1000
    logger.info(
        "hash=%s spec=%s conf=%.3f latency=%.1fms",
        text_hash, routed, top_conf, latency,
    )

    return PredictionResponse(
        specialty=routed,
        confidence=round(top_conf, 4),
        is_confident=confident,
        is_high_acuity=top_spec in HIGH_ACUITY,
        recommendation=rec,
        top_k=top_preds,
        text_hash=text_hash,
        latency_ms=round(latency, 2),
    )


@app.get("/health")
async def health() -> dict:
    """Проверка работоспособности сервиса."""
    return {
        "status":    "ok",
        "model":     MODEL_NAME,
        "n_classes": len(le.classes_),
        "threshold": CONF_THRESHOLD,
        "device":    str(DEVICE),
    }


@app.get("/classes")
async def classes() -> dict:
    """Список всех поддерживаемых специальностей."""
    return {
        "n_classes":       len(le.classes_),
        "specialties":     list(le.classes_),
        "high_acuity":     sorted(HIGH_ACUITY),
        "fallback_class":  FALLBACK_CLASS,
    }
