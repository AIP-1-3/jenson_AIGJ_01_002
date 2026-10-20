"""결과를 MQTT 로 발행 (retain). 발행 실패는 서비스를 멈추지 않는다.

토픽: {PUBLISH_PREFIX}/{공간}  예) tsinfer/mskim/실습실
retain 이라 새로 구독한 쪽도 마지막 결과를 바로 받는다 (과제 8-1).
공용 브로커에 쓰는 것이므로 접두사는 반드시 본인 이름으로 구분한다.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import paho.mqtt.client as mqtt

from tsinfer.config import MqttSettings

log = logging.getLogger("publisher")


def publish_prefix() -> str | None:
    p = (os.getenv("PUBLISH_PREFIX") or "").strip().rstrip("/")
    if not p or "<" in p:
        return None
    return p


class Publisher:
    def __init__(self, settings: MqttSettings, prefix: str):
        self.prefix = prefix
        # 수집기와 같은 client id 로 붙으면 브로커가 수집기를 끊는다 → 접미사를 붙인다
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{settings.client_id}-svc")
        if settings.user:
            self.client.username_pw_set(settings.user, settings.password)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.connect_async(settings.host, settings.port, keepalive=60)
        self.client.loop_start()

    def publish(self, result: dict[str, Any]) -> bool:
        if not self.client.is_connected():
            log.warning("MQTT 미접속 — 이번 주기 발행 건너뜀")
            return False
        ok = True
        for space, body in result["spaces"].items():
            payload = {"schema_version": result["schema_version"], "generated_at": result["generated_at"],
                       "model": result["model"], "space": space, **body}
            info = self.client.publish(f"{self.prefix}/{space}", json.dumps(payload, ensure_ascii=False),
                                       qos=1, retain=True)
            ok &= info.rc == mqtt.MQTT_ERR_SUCCESS
        return ok

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
