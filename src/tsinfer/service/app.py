"""상시 추론 서비스 (과제 8): 매분 추론 → MQTT 발행 → 조회 API.

- 추론 루프는 백그라운드 스레드에서 매분 0초+5초에 한 번 돈다 (수집기가 막 받은 값을 반영).
- 주기 하나가 실패해도 기록만 남기고 다음 주기로 간다. 연속 실패는 /health 에 드러난다.
- 매 주기를 logs/service_cycles.jsonl 에 한 줄씩 남겨 12시간 무중단 보고(summary)에 쓴다.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from tsinfer.config import ROOT
from tsinfer.service.engine import Engine

log = logging.getLogger("service")


class Runner:
    def __init__(self, engine: Engine, publisher=None, cycle_log: Path | None = None, interval_s: int = 60,
                 monitor=None):
        self.engine = engine
        self.publisher = publisher
        self.monitor = monitor          # 자기 채점 (SELF_MONITOR=1 일 때만, 없으면 기존 동작 그대로)
        self.cycle_log = cycle_log or ROOT / "logs" / "service_cycles.jsonl"
        self.interval_s = interval_s
        self.latest: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.consecutive_failures = 0
        self.started_at = datetime.now(timezone.utc)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def cycle(self, now: datetime | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"ts": (now or datetime.now(timezone.utc)).isoformat()}
        try:
            result = self.engine.run_once(now)
            self.latest = result
            self.last_error = None
            self.consecutive_failures = 0
            rec.update(ok=True, model=result["model"],
                       data_ok={s: b["data_ok"] for s, b in result["spaces"].items()},
                       alerts={s: [a["type"] for a in b["anomaly"]["alerts"]] for s, b in result["spaces"].items()})
            if self.publisher is not None:
                try:
                    rec["published"] = self.publisher.publish(result)
                except Exception as e:  # 발행 실패로 추론을 멈추지 않는다
                    log.exception("발행 실패")
                    rec["published"] = False
                    rec["publish_error"] = repr(e)
            if self.monitor is not None:
                try:
                    sent = self.monitor.observe(result)
                    rec["monitor"] = {**self.monitor.snapshot(), "sent": sent}
                except Exception as e:  # 채점 실패로 추론을 멈추지 않는다
                    log.exception("자기 채점 실패")
                    rec["monitor_error"] = repr(e)
        except Exception as e:
            log.exception("추론 주기 실패")
            self.last_error = repr(e)
            self.consecutive_failures += 1
            rec.update(ok=False, error=repr(e))
        rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._write(rec)
        return rec

    def _write(self, rec: dict[str, Any]) -> None:
        try:
            self.cycle_log.parent.mkdir(parents=True, exist_ok=True)
            with self.cycle_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("주기 기록 실패")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.cycle()
            # 다음 정분 +5초까지 대기 (주기가 밀려도 누적되지 않게 벽시계 기준)
            wait = (5 - time.time()) % self.interval_s
            self._stop.wait(wait if wait >= 1 else wait + self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="inference-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self.publisher is not None:
            self.publisher.close()


def create_app(runner: Runner, start_loop: bool = True) -> FastAPI:
    # Starlette 1.x 에서 add_event_handler/on_event 가 없어져 lifespan 으로 시작·종료를 건다
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_loop:
            runner.start()
        try:
            yield
        finally:
            if start_loop:
                runner.stop()

    app = FastAPI(title="tsinfer", version="0.1", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest = runner.latest
        body = {
            "ok": runner.consecutive_failures == 0 and latest is not None,
            "started_at": runner.started_at.isoformat(),
            "last_generated_at": latest["generated_at"] if latest else None,
            "consecutive_failures": runner.consecutive_failures,
            "last_error": runner.last_error,
            "model": latest["model"] if latest else None,
            "data_ok": {s: b["data_ok"] for s, b in latest["spaces"].items()} if latest else {},
        }
        monitor = getattr(runner, "monitor", None)
        if monitor is not None:   # 켜졌을 때만 필드가 생긴다
            body["monitor"] = monitor.snapshot()
        return body

    @app.get("/latest")
    def latest_all() -> dict[str, Any]:
        if runner.latest is None:
            raise HTTPException(503, "아직 첫 추론 전")
        return runner.latest

    @app.get("/latest/{space}")
    def latest_space(space: str) -> dict[str, Any]:
        if runner.latest is None:
            raise HTTPException(503, "아직 첫 추론 전")
        body = runner.latest["spaces"].get(space)
        if body is None:
            raise HTTPException(404, f"없는 공간: {space}")
        return {"generated_at": runner.latest["generated_at"], "model": runner.latest["model"],
                "space": space, **body}

    return app
