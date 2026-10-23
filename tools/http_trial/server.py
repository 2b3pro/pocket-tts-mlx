"""Opt-in, loopback-only HTTP comparison harness; not a production server.

All MLX operations run on the lifespan/event-loop thread. No decoder worker
threads or changes to the production VoiceServer are used.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np

PAUSE = re.compile(r"\[(\d+(?:\.\d+)?)s\]", re.IGNORECASE)
AUDIO_NAME = re.compile(r"pair-\d{2}-[AB]\.wav")
VOICE_NAME = re.compile(r"[A-Za-z0-9_-]{1,100}")


def split_pauses(text: str) -> list[str | float]:
    parts: list[str | float] = []
    start = 0
    for match in PAUSE.finditer(text):
        if text[start:match.start()].strip():
            parts.append(text[start:match.start()].strip())
        seconds = min(10.0, float(match.group(1)))
        if seconds:
            parts.append(seconds)
        start = match.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def level(audio: np.ndarray, target: float) -> np.ndarray:
    """Match the resident server's full-clip RMS scaling and peak ceiling."""
    if not audio.size:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms:
        audio = audio * (10 ** (target / 20) / rms)
    peak = float(np.max(np.abs(audio)))
    return audio * (0.99 / peak) if peak > 0.99 else audio


def pcm16(audio: np.ndarray) -> bytes:
    return np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


@dataclass
class Settings:
    states: Path
    artifacts: Path
    voice: str = "alba"
    model: str = "english_2026-04"
    temperature: float = 0.7
    buffered_batch_size: int = 4


class Engine:
    def __init__(self, model, settings: Settings):
        self.model = model
        self.settings = settings
        self.sample_rate = model.sample_rate
        self.owner = threading.get_ident()
        self.lock = asyncio.Lock()
        self.states = {}
        self.active = 0
        self.completed = 0
        self.cancelled = 0
        self.frames = 0
        self.voice_state(settings.voice)

    def voice_path(self, voice: str) -> Path:
        token = voice.rsplit("/", 1)[-1].removesuffix(".safetensors")
        if not VOICE_NAME.fullmatch(token):
            raise FileNotFoundError("invalid voice token")
        path = self.settings.states / f"{token}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"no state for {token!r}")
        return path

    def voice_state(self, voice: str):
        assert threading.get_ident() == self.owner, "MLX used from a different thread"
        path = self.voice_path(voice)
        if path.name not in self.states:
            self.states[path.name] = self.model.get_state_for_audio_prompt(path)
        return self.states[path.name]

    async def chunks(self, request, parts, voice, temperature, seed, batch):
        async with self.lock:
            assert threading.get_ident() == self.owner
            self.active += 1
            finished = False
            try:
                state = self.voice_state(voice)
                for index, part in enumerate(parts):
                    if await request.is_disconnected():
                        return
                    if isinstance(part, float):
                        yield np.zeros(int(part * self.sample_rate), dtype=np.float32)
                        continue
                    stream = self.model.generate_audio_stream(
                        state, part, temperature=temperature,
                        seed=None if seed is None else seed + index,
                        decode_batch_size=batch, copy_state=True,
                    )
                    try:
                        for chunk in stream:
                            if await request.is_disconnected():
                                return
                            self.frames += 1
                            yield np.asarray(chunk, dtype=np.float32).reshape(-1)
                            await asyncio.sleep(0)
                    finally:
                        stream.close()
                finished = True
            finally:
                self.active -= 1
                if finished:
                    self.completed += 1
                else:
                    self.cancelled += 1


def create_app(settings: Settings, loader=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, Response, StreamingResponse

    # FastAPI resolves annotations in the module namespace.
    globals()["Request"] = Request

    @asynccontextmanager
    async def lifespan(app):
        load = loader
        if load is None:
            from pocket_tts_mlx import TTSModel
            load = TTSModel.load_model
        model = load(settings.model, temp=settings.temperature)
        app.state.engine = Engine(model, settings)
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health():
        engine = app.state.engine
        return {
            "status": "ready", "engine": "mlx-http-trial", "model": settings.model,
            "voices_loaded": len(engine.states), "pinned": [settings.voice],
            "active": engine.active, "completed": engine.completed,
            "cancelled": engine.cancelled, "frames": engine.frames,
        }

    @app.post("/generate")
    async def generate(request: Request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            text = str(body.get("text") or "").strip()
            voice = str(body.get("voice") or "")
            if not text or not voice or len(text) > 20_000:
                raise ValueError("text and voice required; text limit is 20000 characters")
            parts = split_pauses(text)
            if not parts:
                raise ValueError("nothing to speak")
            temperature = float(body.get("temperature", settings.temperature))
            if not math.isfinite(temperature):
                raise ValueError("temperature must be finite")
            temperature = min(1.5, max(0.1, temperature))
            seed = None if body.get("seed") is None else int(body["seed"])
            streaming = body.get("stream") is not False
            batch = 1 if streaming else settings.buffered_batch_size
            target = body.get("targetRmsDb")
            if target is not None:
                target = float(target)
                if not math.isfinite(target) or not -60 <= target <= 0:
                    raise ValueError("targetRmsDb must be between -60 and 0")
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(400, str(exc)) from exc
        engine = app.state.engine
        try:
            engine.voice_path(voice)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        headers = {"x-sample-rate": str(engine.sample_rate), "x-channels": "1", "x-model": settings.model}
        chunks = engine.chunks(request, parts, voice, temperature, seed, batch)
        if not streaming:
            collected = [chunk async for chunk in chunks]
            audio = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
            if target is not None:
                audio = level(audio, target)
            return Response(pcm16(audio), media_type="application/octet-stream", headers=headers)

        async def pcm():
            try:
                async for chunk in chunks:
                    yield pcm16(chunk)
            finally:
                await chunks.aclose()

        return StreamingResponse(pcm(), media_type="application/octet-stream", headers=headers)

    @app.get("/trial")
    async def listening_page():
        return FileResponse(Path(__file__).with_name("listen.html"))

    @app.get("/trial/pairs.json")
    async def listening_pairs():
        path = settings.artifacts / "pairs.json"
        if not path.is_file():
            raise HTTPException(404, "listening clips are not ready")
        return FileResponse(path)

    @app.get("/trial/audio/{name}")
    async def listening_audio(name: str):
        if not AUDIO_NAME.fullmatch(name):
            raise HTTPException(404)
        path = settings.artifacts / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav")

    @app.post("/trial/votes")
    async def save_votes(request: Request):
        body = await request.json()
        pairs = json.loads((settings.artifacts / "pairs.json").read_text())
        allowed = {pair["id"] for pair in pairs}
        if not isinstance(body, dict) or not isinstance(body.get("votes"), dict):
            raise HTTPException(400, "votes must be an object")
        if set(body["votes"]) != allowed or any(v not in ("A", "B", "tie") for v in body["votes"].values()):
            raise HTTPException(400, "choose A, B, or tie for every pair")
        notes = body.get("notes", "")
        if not isinstance(notes, str) or len(notes) > 2000:
            raise HTTPException(400, "notes must be text, at most 2000 characters")
        path = settings.artifacts / "votes.json"
        path.write_text(json.dumps({"votes": body["votes"], "notes": notes, "saved_at": time.time()}, indent=2) + "\n")
        path.chmod(0o600)
        return {"saved": True}

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--model", default="english_2026-04")
    parser.add_argument("--port", default=8893, type=int)
    args = parser.parse_args()
    settings = Settings(states=args.states, artifacts=args.artifacts, voice=args.voice, model=args.model)
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
