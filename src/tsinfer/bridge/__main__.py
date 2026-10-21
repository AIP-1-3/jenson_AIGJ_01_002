"""추론 결과 → InfluxDB 브리지: python -m tsinfer.bridge [--dry-run]

서비스(tsinfer-service)가 MQTT 로 내는 `{PUBLISH_PREFIX}/+` 결과를 받아 InfluxDB 에 쓴다.
서비스·수집기와 별개 프로세스라, 이 브리지가 죽거나 InfluxDB 가 안 닿아도 추론과 수집은 그대로 돈다.

- 쓰기 대상: INFLUX_RESULT_URL / INFLUX_RESULT_ORG (없으면 INFLUX_URL / INFLUX_ORG), 버킷 INFLUX_RESULT_BUCKET(기본 tsinfer),
  토큰 INFLUX_WRITE_TOKEN(없으면 INFLUX_TOKEN). 원본 버킷에는 쓰지 않는다.
  과거 데이터를 받는 서버(INFLUX_URL)와 Grafana 가 보는 학교 서버가 다를 수 있어 따로 둔다.
- 쓰기 실패분은 메모리에 최대 MAX_PENDING 점까지 들고 있다가 다음 메시지 때 같이 보낸다.
- --dry-run: InfluxDB 에 쓰지 않고 line protocol 을 화면에 찍는다 (접속 정보 없이 형식 확인용).
- --check: 점 1개(tsinfer_bridge_check)를 써 보고 끝낸다. 주소·조직·토큰·버킷 권한 확인용.
- BRIDGE_MIRROR_SENSORS=1: 수집기 SQLite 센서값도 결과 서버의 INFLUX_MIRROR_BUCKET(기본 iot-bucket)에 복사한다.
  원격 InfluxDB 에 닿지 않아 Jetson 안의 InfluxDB 를 쓸 때만 켠다 (mirror.py). 원본 서버(INFLUX_URL)에는 절대 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
from collections import deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from tsinfer.bridge.points import Point, to_points
from tsinfer.config import ROOT, mqtt_settings
from tsinfer.service.publisher import publish_prefix

log = logging.getLogger("bridge")
MAX_PENDING = 5000      # 약 1시간 분량 (공간 4개 × 분당 ~20점). 넘으면 오래된 것부터 버린다.


def to_line(p: Point) -> str:
    from influxdb_client import Point as IPoint, WritePrecision

    ip = IPoint(p.measurement).time(p.time, WritePrecision.S)
    for k, v in p.tags.items():
        ip = ip.tag(k, v)
    for k, v in p.fields.items():
        ip = ip.field(k, v)
    return ip.to_line_protocol()


class InfluxSink:
    def __init__(self, bucket: str | None = None, max_pending: int = MAX_PENDING):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS

        token = os.getenv("INFLUX_WRITE_TOKEN") or os.environ["INFLUX_TOKEN"]
        self.org = os.getenv("INFLUX_RESULT_ORG") or os.environ["INFLUX_ORG"]
        self.bucket = bucket or os.getenv("INFLUX_RESULT_BUCKET") or "tsinfer"
        if bucket is None and self.bucket in {os.getenv("INFLUX_BUCKET") or "iot-bucket", "sensor_data"}:
            raise RuntimeError("INFLUX_RESULT_BUCKET 이 원본 버킷과 같습니다. 결과는 별도 버킷에 씁니다.")
        self.client = InfluxDBClient(url=os.getenv("INFLUX_RESULT_URL") or os.environ["INFLUX_URL"], token=token, org=self.org, timeout=10_000)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.pending: deque[str] = deque(maxlen=max_pending)

    def check(self) -> None:
        """실패하면 예외를 그대로 올려 원인(401 토큰, 404 버킷 등)을 화면에 보인다."""
        line = f"tsinfer_bridge_check ok=1i {int(datetime.now(timezone.utc).timestamp())}"
        self.write_api.write(bucket=self.bucket, org=self.org, record=line, write_precision="s")

    def write(self, lines: list[str]) -> None:
        self.pending.extend(lines)
        batch = list(self.pending)
        try:
            for i in range(0, len(batch), 5000):     # 요청 하나가 너무 커지지 않게
                self.write_api.write(bucket=self.bucket, org=self.org, record=batch[i:i + 5000])
        except Exception as e:  # 네트워크·권한 오류로 브리지가 죽지 않게 한다
            log.warning("InfluxDB 쓰기 실패(%s) — %d점 보류", type(e).__name__, len(self.pending))
            return
        self.pending.clear()

    def close(self) -> None:
        self.client.close()


def mirror_enabled() -> bool:
    if (os.getenv("BRIDGE_MIRROR_SENSORS") or "").strip() not in ("1", "true", "yes"):
        return False
    target = (os.getenv("INFLUX_RESULT_URL") or "").rstrip("/")
    if not target or target == (os.getenv("INFLUX_URL") or "").rstrip("/"):
        # 원본 서버에 센서값을 다시 쓰면 공용 데이터가 중복된다
        raise RuntimeError("BRIDGE_MIRROR_SENSORS 는 INFLUX_RESULT_URL 이 Jetson 안의 InfluxDB 일 때만 켭니다.")
    return True


def mirror_sink() -> InfluxSink:
    return InfluxSink(bucket=os.getenv("INFLUX_MIRROR_BUCKET") or "iot-bucket", max_pending=200_000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 line protocol 출력")
    ap.add_argument("--check", action="store_true", help="InfluxDB 에 점 1개를 써 보고 끝냄")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.check:
        load_dotenv(ROOT / ".env")
        sink = InfluxSink()
        sinks = [sink] + ([mirror_sink()] if mirror_enabled() else [])
        for sk in sinks:
            sk.check()
            print(f"쓰기 성공: {sk.client.url} org={sk.org} bucket={sk.bucket}")
            sk.close()
        return 0

    settings = mqtt_settings()          # .env 도 여기서 읽힌다
    prefix = publish_prefix()
    if not prefix:
        raise RuntimeError("PUBLISH_PREFIX 가 비어 있습니다. 서비스와 같은 값(예: tsinfer/<이름>)을 넣으세요.")
    sink = None if args.dry_run else InfluxSink()
    topic = f"{prefix}/+"
    mirror = None
    if sink is not None and mirror_enabled():
        from tsinfer.bridge.mirror import SensorMirror
        from tsinfer.config import db_path

        mirror = SensorMirror(db_path(), mirror_sink(),
                              backfill_hours=float(os.getenv("BRIDGE_MIRROR_BACKFILL_HOURS") or 24))
        mirror.start()
        log.info("센서 복사 켬: %s → bucket=%s", db_path(), mirror.sink.bucket)

    # 수집기·서비스와 같은 client id 면 브로커가 한쪽을 끊는다 → 접미사
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{settings.client_id}-bridge")
    if settings.user:
        client.username_pw_set(settings.user, settings.password)
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(c, _u, _f, reason_code, _p):
        if reason_code.is_failure:
            log.error("접속 실패: %s", reason_code)
            return
        c.subscribe(topic, qos=1)
        log.info("구독: %s → %s", topic, "dry-run" if sink is None else f"bucket={sink.bucket}")

    def on_message(_c, _u, msg):
        try:
            lines = [to_line(p) for p in to_points(json.loads(msg.payload))]
        except Exception:  # 형식이 다른 메시지 하나 때문에 멈추지 않는다
            log.exception("변환 실패: topic=%s", msg.topic)
            return
        if sink is None:
            print("\n".join(lines), flush=True)
        else:
            sink.write(lines)

    client.on_connect = on_connect
    client.on_message = on_message
    signal.signal(signal.SIGTERM, lambda *_: client.disconnect())
    signal.signal(signal.SIGINT, lambda *_: client.disconnect())

    log.info("브리지 시작 (%s)", ROOT)
    client.connect_async(settings.host, settings.port, keepalive=60)
    client.loop_forever(retry_first_connection=True)
    if mirror is not None:
        mirror.stop()
    if sink is not None:
        sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
