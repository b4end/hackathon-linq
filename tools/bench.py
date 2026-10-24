"""Замеры latency (обязательное требование кейса): одна картинка, N прогонов, статистика.
Запуск: python -m tools.bench --image /путь/картинка.jpg --runs 20
JSON落ается в logs/; позже эти цифры пойдут в GET /api/performance."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from linq.backend.cloud import CloudBackend
from linq.config import Settings
from service.pipeline import Pipeline


def make_backend(name: str):
    if name == 'cloud':
        return CloudBackend(Settings())
    if name == 'mock':
        from tests.mock_backend import MockBackend  # только dry-run без VPN
        return MockBackend()
    raise SystemExit(f'неизвестный бэкенд {name!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Бенчмарк инференса LinQ')
    parser.add_argument('--image', required=True)
    parser.add_argument('--program', default='resnet50')
    parser.add_argument('--backend', choices=('cloud', 'mock'), default='cloud')
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--out', type=Path, default=None,
                        help='куда сохранить JSON (по умолчанию logs/bench_<время>.json)')
    args = parser.parse_args()

    backend = make_backend(args.backend)
    pipeline = Pipeline(backend)
    try:
        for _ in range(args.warmup):
            pipeline.infer_image(args.image, args.program)
        latencies = []
        started = time.monotonic()
        for _ in range(args.runs):
            result = pipeline.infer_image(args.image, args.program)
            latencies.append(result.latency_s)
        total = time.monotonic() - started
    finally:
        backend.close()

    arr = np.array(latencies)
    report = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'program': args.program,
        'backend': args.backend,
        'image': str(args.image),
        'warmup': args.warmup,
        'runs': args.runs,
        'latency_mean_s': float(arr.mean()),
        'latency_p50_s': float(np.percentile(arr, 50)),
        'latency_p95_s': float(np.percentile(arr, 95)),
        'fps': float(args.runs / total),
        'total_s': float(total),
        'worker': getattr(backend, 'last_worker', None),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = args.out or Path('logs') / f'bench_{datetime.now():%Y%m%d_%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'сохранено: {out}')


if __name__ == '__main__':
    main()