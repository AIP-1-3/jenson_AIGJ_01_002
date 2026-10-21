"""Grafana 알림 provisioning 이 대시보드·브리지와 같은 이름을 쓰는지, Grafana 가 읽을 수 있는 모양인지 검사한다.

실제 Grafana 11.3 + InfluxDB 2.7 로 네 규칙이 모두 발화하는 것은 2026-10-21 컨테이너로 따로 확인했다 (docs/grafana.md 알림).
"""
from pathlib import Path

import yaml

from test_bridge import _payload
from tsinfer.bridge.points import to_points

ROOT = Path(__file__).resolve().parents[1] / "grafana" / "provisioning"
ALERT = ROOT / "alerting"


def _rules():
    return [r for g in yaml.safe_load((ALERT / "rules.yaml").read_text(encoding="utf-8"))["groups"] for r in g["rules"]]


def test_rules_use_provisioned_datasource_and_wide_frames():
    ds_uid = yaml.safe_load((ROOT / "datasources" / "influx.yaml").read_text(encoding="utf-8"))["datasources"][0]["uid"]
    rules = _rules()
    assert {r["uid"] for r in rules} == {"tsinfer-vent-forecast", "tsinfer-anomaly", "tsinfer-sensor-gap", "tsinfer-service-silent"}
    for r in rules:
        q = next(d for d in r["data"] if d["refId"] == "A")
        assert q["datasourceUid"] == ds_uid
        # Grafana 알림은 wide 시계열만 받는다. keep 이 없으면 'input data must be a wide series' 오류 (시험에서 실제로 남)
        assert "|> keep(columns:" in q["model"]["query"].strip().splitlines()[-1]
        assert r["condition"] == "C"


def test_rules_read_fields_the_bridge_writes():
    pts = to_points(_payload())
    result_fields = set(pts[0].fields)
    text = (ALERT / "rules.yaml").read_text(encoding="utf-8")
    assert "fc_co2_60" in result_fields and '"fc_co2_60"' in text
    assert "data_ok" in result_fields and '"data_ok"' in text
    assert '"tsinfer_alert"' in text and '"tsinfer_result"' in text


def test_no_dollar_except_env_placeholders():
    """provisioning 파일의 $ 는 Grafana 가 환경변수로 바꾼다. 규칙 템플릿에 $ 가 섞이면 조용히 빈 값이 된다."""
    assert "$" not in (ALERT / "rules.yaml").read_text(encoding="utf-8")
    body = [ln for ln in (ALERT / "contact-points.yaml").read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    assert [ln.strip() for ln in body if "$" in ln] == ["url: ${DISCORD_WEBHOOK_URL}"]


def test_policy_routes_to_the_contact_point():
    cp = yaml.safe_load((ALERT / "contact-points.yaml").read_text(encoding="utf-8"))["contactPoints"][0]["name"]
    pol = yaml.safe_load((ALERT / "policies.yaml").read_text(encoding="utf-8"))["policies"][0]
    assert pol["receiver"] == cp
