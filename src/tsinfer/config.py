"""설정 로딩: 비밀 값은 .env(환경변수), 실험·매핑 값은 configs/*.yaml.

학습·평가·서비스가 모두 이 모듈로 설정을 읽어서 전처리 조건이 한곳에서 관리되게 한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class MqttSettings:
    host: str
    port: int
    user: str | None
    password: str | None
    client_id: str


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"{key} 가 비어 있습니다. .env 를 확인하세요 (.env.example 참고).")
    return value


def mqtt_settings() -> MqttSettings:
    load_dotenv(ROOT / ".env")
    client_id = _require("MQTT_CLIENT_ID")
    if "<" in client_id:
        raise RuntimeError("MQTT_CLIENT_ID 의 <이름> 을 본인 이름으로 바꾸세요.")
    return MqttSettings(
        host=_require("MQTT_HOST"),
        port=int(os.getenv("MQTT_PORT", "1883")),
        user=os.getenv("MQTT_USER") or None,
        password=os.getenv("MQTT_PASS") or None,
        client_id=client_id,
    )


def db_path() -> Path:
    load_dotenv(ROOT / ".env")
    path = Path(os.getenv("DB_PATH", "data/tsinfer.db"))
    return path if path.is_absolute() else ROOT / path
