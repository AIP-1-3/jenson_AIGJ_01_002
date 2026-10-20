from datetime import datetime, timezone

from tsinfer.ingest.influx_export import chunks


def test_chunks_cover_range_without_gaps():
    a = datetime(2026, 7, 22, tzinfo=timezone.utc)
    b = datetime(2026, 8, 10, tzinfo=timezone.utc)
    parts = list(chunks(a, b, 7))
    assert parts[0][0] == a and parts[-1][1] == b
    assert all(x[1] == y[0] for x, y in zip(parts, parts[1:]))
    assert len(parts) == 3


class _Rec:
    def __init__(self, t, v, **tags):
        self._t, self._v, self.values = t, v, tags

    def get_time(self):
        return self._t

    def get_value(self):
        return self._v


class _Client:
    def __init__(self, recs):
        self.recs = recs

    def query_api(self):
        return self

    def query_stream(self, flux, org):
        assert 'r._measurement == "activity"' in flux
        return iter(self.recs)


def test_fetch_renames_metric_and_drops_non_numeric():
    from tsinfer.ingest.influx_export import COLUMNS, fetch

    t = datetime(2026, 8, 1, tzinfo=timezone.utc)
    recs = [_Rec(t, 1, location="실습실", device_id="d1", point="p"), _Rec(t, "x", location="실습실", device_id="d1")]
    df = fetch(_Client(recs), "org", "iot-bucket", "activity", t, t)
    assert list(df.columns) == COLUMNS
    assert df["metric"].tolist() == ["pir_activity"]
    assert df["value"].tolist() == [1.0]
