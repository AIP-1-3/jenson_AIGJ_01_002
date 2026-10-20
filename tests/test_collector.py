from datetime import datetime, timezone

from tsinfer.collector.mapping import Mapper
from tsinfer.collector.store import Store

CONFIG = {
    "sources": [
        {
            "name": "env_v1",
            "topic": "sensors/{space}/env",
            "payload": {"co2": "co2", "temperature": "temp"},
            "timestamp": {"field": "ts", "format": "epoch_ms"},
        }
    ]
}
NOW = datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc)


def test_subscription_uses_wildcards():
    assert Mapper(CONFIG).subscriptions == ["sensors/+/env"]


def test_parse_maps_fields_and_space():
    out = Mapper(CONFIG).parse("sensors/lounge/env", b'{"co2": 612, "temp": "23.5", "ts": 1760918400000}', NOW)
    assert {(r.space, r.metric, r.value) for r in out} == {("lounge", "co2", 612.0), ("lounge", "temperature", 23.5)}
    assert out[0].sensor_ts == datetime.fromtimestamp(1760918400, tz=timezone.utc)


def test_parse_ignores_unknown_topic_and_broken_payload():
    m = Mapper(CONFIG)
    assert m.parse("other/topic", b"{}", NOW) == []
    assert m.parse("sensors/lounge/env", b"not json", NOW) == []


def test_missing_sensor_ts_falls_back_to_received():
    out = Mapper(CONFIG).parse("sensors/lounge/env", b'{"co2": 500}', NOW)
    assert out[0].sensor_ts is None


def test_store_dedups_retransmission(tmp_path):
    m = Mapper(CONFIG)
    store = Store(tmp_path / "t.db")
    msg = b'{"co2": 612, "ts": 1760918400000}'
    assert store.write(m.parse("sensors/lounge/env", msg, NOW)) == 1
    assert store.write(m.parse("sensors/lounge/env", msg, NOW)) == 0


# --- 실제 브로커 형식 (2026-10-20 확인) ---
from tsinfer.config import load_yaml

CHIRP = (
    b'{"deduplicationId":"f89b","time":"2026-10-20T01:38:19.287+00:00",'
    b'"deviceInfo":{"deviceName":"AM103-081175","devEui":"24e124725d081175","tags":{"location":"\xec\x82\xac\xeb\xac\xb4\xec\x8b\xa4"}},'
    b'"object":{"co2":612,"temperature":23.4,"humidity":41.5}}'
)
XFLOW = (
    b'{"ts":1792459940616,"values":{"\xea\xb0\x9c\xeb\xb0\x9c\xec\x8b\xa4-\xec\x8b\xa4\xeb\x82\xb4\xea\xb8\xb0":'
    b'{"current_humidity":48,"current_temperature":22.3,"power":false,"target_temperature":18}}}'
)


def test_repo_sensors_yaml_parses_chirpstack_uplink():
    out = Mapper(load_yaml("sensors.yaml")).parse(
        "application/96b4/device/24e124725d081175/event/up", CHIRP, NOW
    )
    assert {(r.space, r.device, r.metric, r.value) for r in out} == {
        ("사무실", "AM103-081175", "co2", 612.0),
        ("사무실", "AM103-081175", "temperature", 23.4),
        ("사무실", "AM103-081175", "humidity", 41.5),
    }
    assert out[0].sensor_ts.isoformat() == "2026-10-20T01:38:19.287000+00:00"


def test_repo_sensors_yaml_parses_xflow_idu_per_room():
    out = Mapper(load_yaml("sensors.yaml")).parse("xflow/hvacr/idu/5442/status", XFLOW, NOW)
    got = {(r.space, r.metric, r.value) for r in out}
    assert ("개발실-실내기", "hvac_power", 0.0) in got
    assert {r.device for r in out} == {"5442"}
    assert ("개발실-실내기", "hvac_current_temperature", 22.3) in got


def test_space_map_and_drop_unmapped():
    cfg = {"sources": [{
        "name": "c", "topic": "application/{app}/device/{eui}/event/up",
        "space": {"from": "eui", "map": {"24e124725d081175": "lounge"}, "unmapped": "drop"},
        "payload": {"co2": "object.co2"},
    }]}
    m = Mapper(cfg)
    assert m.parse("application/a/device/24e124725d081175/event/up", CHIRP, NOW)[0].space == "lounge"
    assert m.parse("application/a/device/ffff/event/up", CHIRP, NOW) == []


def test_two_sensors_in_same_space_are_kept_apart(tmp_path):
    m = Mapper(load_yaml("sensors.yaml"))
    store = Store(tmp_path / "t.db")
    for name in ("EM500-CO2-152590", "EM500-CO2-152862"):
        body = ('{"time":"2026-10-20T01:38:00+00:00","deviceInfo":{"deviceName":"%s",'
                '"tags":{"location":"lab"}},"object":{"co2":900}}' % name).encode()
        assert store.write(m.parse("application/a/device/x/event/up", body, NOW)) == 1
