"""GLiNER2 API compartilhada — extração estruturada para todos os projetos FabriaIA."""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

APP_NAME = "gliner-api"
MODEL_ID = os.getenv("GLINER_MODEL", "fastino/gliner2.5-multi-v1").strip()
API_TOKEN = (os.getenv("GLINER_API_TOKEN") or "").strip()
MAP_LOCATION = os.getenv("GLINER_DEVICE", "cpu").strip() or "cpu"

app = FastAPI(
    title="GLiNER API",
    description="Schema-based information extraction (GLiNER2.5) — serviço compartilhado.",
    version="1.0.0",
)

_model = None
_model_error: str | None = None
_model_lock = threading.Lock()
_loaded_at: float | None = None


def _exigir_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization:
        raise HTTPException(401, "Authorization Bearer obrigatório")
    raw = authorization.strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    if raw != API_TOKEN:
        raise HTTPException(403, "token inválido")


def get_model():
    global _model, _model_error, _loaded_at
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error:
            raise HTTPException(503, f"modelo indisponível: {_model_error}")
        try:
            from gliner2 import AutoExtractor

            t0 = time.time()
            _model = AutoExtractor.from_pretrained(MODEL_ID, map_location=MAP_LOCATION)
            _loaded_at = time.time()
            app.state.load_seconds = round(_loaded_at - t0, 2)
        except Exception as exc:  # noqa: BLE001
            _model_error = str(exc)[:500]
            raise HTTPException(503, f"falha ao carregar modelo: {_model_error}") from exc
        return _model


class EntitiesRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    labels: list[str] | dict[str, str] = Field(
        ...,
        description="Lista de labels ou mapa label→descrição",
    )
    include_spans: bool = False


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    labels: list[str] | dict[str, str]
    multi_label: bool = True


class ExtractRequest(BaseModel):
    """Extração combinada: entidades e/ou classificação num pedido."""

    text: str = Field(..., min_length=1, max_length=200_000)
    entities: list[str] | dict[str, str] | None = None
    classify: list[str] | dict[str, str] | None = None
    multi_label: bool = True
    include_spans: bool = False


@app.on_event("startup")
def _warmup() -> None:
    """Pré-carrega o modelo em background para o 1º request não esperar o download."""
    if os.getenv("GLINER_WARMUP", "1").strip() in ("0", "false", "no"):
        return

    def _run() -> None:
        try:
            get_model()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_run, daemon=True, name="gliner-warmup").start()


@app.get("/health")
def health() -> dict[str, Any]:
    ready = _model is not None
    return {
        "status": "ok" if ready or _model_error is None else "degraded",
        "service": APP_NAME,
        "model": MODEL_ID,
        "device": MAP_LOCATION,
        "ready": ready,
        "error": _model_error,
        "loaded_at": _loaded_at,
        "load_seconds": getattr(app.state, "load_seconds", None),
    }


@app.get("/v1/info")
def info(_: None = Depends(_exigir_token)) -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "model": MODEL_ID,
        "device": MAP_LOCATION,
        "ready": _model is not None,
        "endpoints": [
            "POST /v1/extract/entities",
            "POST /v1/classify",
            "POST /v1/extract",
        ],
    }


@app.post("/v1/extract/entities")
def extract_entities(
    body: EntitiesRequest,
    _: None = Depends(_exigir_token),
) -> dict[str, Any]:
    model = get_model()
    t0 = time.time()
    try:
        result = model.extract_entities(
            body.text,
            body.labels,
            include_spans=body.include_spans,
        )
    except TypeError:
        # checkpoints/API sem include_spans
        result = model.extract_entities(body.text, body.labels)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"extração falhou: {exc}") from exc
    return {
        "status": "ok",
        "model": MODEL_ID,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "result": result if isinstance(result, dict) else {"raw": result},
    }


@app.post("/v1/classify")
def classify(
    body: ClassifyRequest,
    _: None = Depends(_exigir_token),
) -> dict[str, Any]:
    model = get_model()
    t0 = time.time()
    try:
        if hasattr(model, "classify"):
            result = model.classify(
                body.text,
                body.labels,
                multi_label=body.multi_label,
            )
        elif hasattr(model, "predict_classification"):
            result = model.predict_classification(body.text, body.labels)
        else:
            raise HTTPException(501, "checkpoint sem cabeça de classificação")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"classificação falhou: {exc}") from exc
    return {
        "status": "ok",
        "model": MODEL_ID,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "result": result if isinstance(result, dict) else {"labels": result},
    }


@app.post("/v1/extract")
def extract(
    body: ExtractRequest,
    _: None = Depends(_exigir_token),
) -> dict[str, Any]:
    if not body.entities and not body.classify:
        raise HTTPException(400, "informe entities= e/ou classify=")
    model = get_model()
    t0 = time.time()
    out: dict[str, Any] = {}
    try:
        if body.entities:
            try:
                out["entities"] = model.extract_entities(
                    body.text,
                    body.entities,
                    include_spans=body.include_spans,
                )
            except TypeError:
                out["entities"] = model.extract_entities(body.text, body.entities)
        if body.classify:
            if hasattr(model, "classify"):
                out["classification"] = model.classify(
                    body.text,
                    body.classify,
                    multi_label=body.multi_label,
                )
            else:
                out["classification"] = {"aviso": "classificação indisponível neste checkpoint"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"extract falhou: {exc}") from exc
    return {
        "status": "ok",
        "model": MODEL_ID,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "result": out,
    }
