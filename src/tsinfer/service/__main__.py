"""python -m tsinfer.service [--model trend|persistence] [--no-publish] [--once]"""
from __future__ import annotations

import argparse
import json
import logging
import os

from dotenv import load_dotenv

from tsinfer.config import ROOT, db_path, load_yaml, mqtt_settings
from tsinfer.service.app import Runner, create_app
from tsinfer.service.engine import Engine, EngineConfig
from tsinfer.service.predictors import load_forecaster
from tsinfer.service.publisher import Publisher, publish_prefix

log = logging.getLogger("service")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trend(기본) | persistence")
    ap.add_argument("--no-publish", action="store_true", help="MQTT 발행 없이 API 만")
    ap.add_argument("--once", action="store_true", help="한 주기만 돌려 결과를 출력하고 끝낸다")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(ROOT / ".env")

    cfg = load_yaml("default.yaml")
    engine = Engine(cfg, EngineConfig(db_path=db_path()), load_forecaster(cfg, args.model or os.getenv("SERVICE_MODEL")))

    if args.once:
        print(json.dumps(engine.run_once(), ensure_ascii=False, indent=2))
        return 0

    publisher = None
    prefix = publish_prefix()
    if args.no_publish or prefix is None:
        log.warning("MQTT 발행 꺼짐 (--no-publish 또는 .env 의 PUBLISH_PREFIX 비어 있음)")
    else:
        publisher = Publisher(mqtt_settings(), prefix)
        log.info("발행 토픽: %s/<공간> (retain)", prefix)

    import uvicorn

    port = int(os.getenv("API_PORT", "8100"))
    uvicorn.run(create_app(Runner(engine, publisher)), host="0.0.0.0", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
