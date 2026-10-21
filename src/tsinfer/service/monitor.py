"""과제 8 심화 (b): 서비스 자기 채점. 환경변수 SELF_MONITOR=1 일 때만 켜진다.

- 매 주기 공간별 CO₂ 60분 예측을 보관했다가, 60분 뒤 그 5분 칸이 완성되면 엔진이 낸 current.co2
  (엔진이 DB 에서 학습과 같은 전처리로 만든 값)와 비교해 채점한다. DB 를 따로 읽지 않는다.
- 공간별 최근 N개 MAE, 같은 시점 '현재값 유지'(예측 시점의 current.co2) MAE, 그 비율을 낸다.
- 비율이 임계값을 넘거나(그리고 MAE 가 최소 크기 이상) MAE 가 절대 임계값을 넘으면 알린다.
  같은 공간의 알림은 재알림 간격(기본 1시간) 안에 반복하지 않고, 정상으로 돌아오면 한 번 알린다.
- 알림: .env 의 DISCORD_WEBHOOK_URL 로 {"content": ...} POST. 비어 있으면 알림만 건너뛴다.
  전송 실패는 기록만 하고 서비스는 계속 돈다 (타임아웃 3초).
- 시계: 결과의 generated_at 을 쓴다 (테스트에서 가짜 시계를 그대로 넣을 수 있다).

기준 시점 근사: 엔진의 마지막 완성 칸 = floor(generated_at, 5분) − 5분 으로 본다 (엔진과 같은 식).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

log = logging.getLogger("monitor")


class DiscordNotifier:
    def __init__(self, url: str, timeout: float = 3.0):
        self.url = url
        self.timeout = timeout

    def __call__(self, text: str) -> bool:
        req = urllib.request.Request(self.url, data=json.dumps({"content": text[:1900]}).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "User-Agent": "tsinfer-monitor"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return 200 <= r.status < 300
        except Exception as e:   # 알림 실패로 서비스를 멈추지 않는다
            log.warning("디스코드 알림 실패: %r", e)
            return False


class SelfMonitor:
    def __init__(self, cfg: dict[str, Any], notify: Callable[[str], bool] | None = None,
                 horizon_min: int = 60, window_n: int = 60, min_n: int = 12,
                 ratio_threshold: float = 1.2, min_mae_ppm: float = 30.0, mae_threshold_ppm: float = 150.0,
                 realert: timedelta = timedelta(hours=1)):
        self.step = pd.Timedelta(cfg["preprocess"]["resample"])
        self.horizon = horizon_min
        self.notify = notify
        self.window_n, self.min_n = window_n, min_n
        self.ratio_threshold, self.min_mae, self.mae_threshold = ratio_threshold, min_mae_ppm, mae_threshold_ppm
        self.realert = realert
        self.pending: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}   # (공간, 목표 칸) → (예측, 현재값)
        self.errors: dict[str, deque] = {}
        self.alerting: dict[str, bool] = {}
        self.last_alert: dict[str, datetime] = {}
        self.alerts_sent = 0
        self.expired = 0

    @classmethod
    def from_env(cls, cfg: dict[str, Any]) -> "SelfMonitor | None":
        """SELF_MONITOR=1 이 아니면 None (서비스 동작 그대로)."""
        if os.getenv("SELF_MONITOR", "").strip() not in ("1", "true", "yes"):
            return None
        url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if not url:
            log.warning("DISCORD_WEBHOOK_URL 비어 있음 — 자기 채점은 하되 알림은 건너뜁니다")
        return cls(cfg, notify=DiscordNotifier(url) if url else None)

    def _base(self, now: datetime) -> pd.Timestamp:
        return pd.Timestamp(now).floor(self.step) - self.step

    def observe(self, result: dict[str, Any]) -> int:
        """한 주기 결과로 채점·보관·알림. 이번에 보낸 알림 수를 돌려준다."""
        now = datetime.fromisoformat(result["generated_at"])
        base = self._base(now)
        h = pd.Timedelta(minutes=self.horizon)
        for space, body in result["spaces"].items():
            cur = (body.get("current") or {}).get("co2") if body.get("data_ok") else None
            # 1) 채점: 이번 기준 칸이 목표였던 예측
            key = (space, base)
            if key in self.pending and cur is not None:
                pred, persist = self.pending.pop(key)
                self.errors.setdefault(space, deque(maxlen=self.window_n)).append((abs(pred - cur), abs(persist - cur)))
            # 2) 보관: 이번 60분 예측
            pred = ((body.get("forecast") or {}).get("co2") or {}).get(str(self.horizon))
            if pred is not None and cur is not None:
                self.pending.setdefault((space, base + h), (float(pred), float(cur)))
        # 목표 칸이 지나도록(데이터 끊김 등) 못 채점한 예측은 버린다
        old = [k for k in self.pending if k[1] < base - 2 * self.step]
        for k in old:
            del self.pending[k]
        self.expired += len(old)
        return self._check(now)

    def stats(self, space: str) -> dict[str, Any]:
        e = self.errors.get(space)
        if not e:
            return {"n": 0, "mae": None, "mae_persistence": None, "ratio": None}
        mae = sum(a for a, _ in e) / len(e)
        mp = sum(b for _, b in e) / len(e)
        return {"n": len(e), "mae": round(mae, 1), "mae_persistence": round(mp, 1),
                "ratio": round(mae / mp, 2) if mp > 0 else None}

    def _bad(self, s: dict[str, Any]) -> str | None:
        if s["n"] < self.min_n:
            return None
        if s["mae"] >= self.mae_threshold:
            return f"MAE {s['mae']}ppm ≥ {self.mae_threshold:g}"
        if s["ratio"] is not None and s["ratio"] > self.ratio_threshold and s["mae"] >= self.min_mae:
            return f"현재값 유지 대비 {s['ratio']}배 (> {self.ratio_threshold:g})"
        return None

    def _check(self, now: datetime) -> int:
        sent = 0
        for space in self.errors:
            s = self.stats(space)
            why = self._bad(s)
            was = self.alerting.get(space, False)
            msg = None
            if why and (not was or now - self.last_alert.get(space, datetime.min.replace(tzinfo=now.tzinfo)) >= self.realert):
                msg = (f"[tsinfer] {space} CO₂ {self.horizon}분 예측 오차 경고: {why} "
                       f"(최근 {s['n']}개 MAE {s['mae']}, 현재값 유지 {s['mae_persistence']})")
            elif not why and was:
                msg = f"[tsinfer] {space} 예측 오차 정상 복귀 (최근 {s['n']}개 MAE {s['mae']}, 비율 {s['ratio']})"
            self.alerting[space] = bool(why)
            if msg:
                self.last_alert[space] = now
                log.warning(msg)
                if self.notify is not None and self.notify(msg):
                    self.alerts_sent += 1
                    sent += 1
        return sent

    def snapshot(self) -> dict[str, Any]:
        return {
            "horizon_min": self.horizon,
            "pending": len(self.pending),
            "expired": self.expired,
            "alerts_sent": self.alerts_sent,
            "notify": self.notify is not None,
            "spaces": {sp: {**self.stats(sp), "alerting": self.alerting.get(sp, False)} for sp in self.errors},
        }
