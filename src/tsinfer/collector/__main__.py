"""상시 수집기: python -m tsinfer.collector

- 브로커가 재시작되거나 네트워크가 끊겨도 paho 의 자동 재접속(1~30초 간격)으로 다시 붙는다.
- 재접속 때마다 on_connect 에서 다시 구독한다 (clean session 이면 구독이 사라지기 때문).
- 보드 재부팅 뒤 자동 시작은 deploy/systemd 의 user 서비스가 맡는다.
"""
from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from tsinfer.config import db_path, load_yaml, mqtt_settings
from tsinfer.collector.mapping import Mapper
from tsinfer.collector.store import Store

log = logging.getLogger("collector")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = mqtt_settings()
    mapper = Mapper(load_yaml("sensors.yaml"))
    store = Store(db_path())

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=settings.client_id)
    if settings.user:
        client.username_pw_set(settings.user, settings.password)
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(c, _userdata, _flags, reason_code, _props):
        if reason_code.is_failure:
            log.error("접속 실패: %s", reason_code)
            return
        for topic in mapper.subscriptions:
            c.subscribe(topic, qos=1)
        log.info("접속·구독 완료: %s", mapper.subscriptions)

    def on_disconnect(_c, _userdata, _flags, reason_code, _props):
        log.warning("접속 끊김(%s) — 자동 재접속 대기", reason_code)

    def on_message(_c, _userdata, msg):
        try:
            readings = mapper.parse(msg.topic, msg.payload, datetime.now(timezone.utc))
            store.write(readings)
        except Exception:  # 메시지 하나 때문에 수집기가 죽으면 안 된다
            log.exception("메시지 처리 실패: topic=%s", msg.topic)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    def stop(*_):
        client.disconnect()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    client.connect_async(settings.host, settings.port, keepalive=30)
    client.loop_forever(retry_first_connection=True)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
