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
