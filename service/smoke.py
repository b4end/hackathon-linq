"""Проверка тракта VPS -> VPN -> RabbitMQ -> TPU -> ответ.
Запуск на VPS (с VPN):  python -m service.smoke --image /путь/ILSVRC2012_val_00000045.JPEG
Ожидание на их тестовой картинке: класс 257.
Оффлайн (без VPN) пайплайн проверяется через pytest -q."""
import argparse
import sys

from linq.backend.cloud import CloudBackend
from linq.config import Settings
from service.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke-тест облачного тракта LinQ')
    parser.add_argument('--image', required=True, help='путь к картинке на этой машине')
    parser.add_argument('--program', default='resnet50')
    parser.add_argument('--expect-class', type=int, default=257)
    args = parser.parse_args()

    backend = CloudBackend(Settings())
    try:
        result = Pipeline(backend).infer_image(args.image, args.program)
    finally:
        backend.close()

    print(f'backend={result.backend} program={result.program} '
          f'latency={result.latency_s:.2f}s out_shapes={result.out_shapes}')
    if backend.last_worker:
        print(f'worker pid={backend.last_worker.get("worker_pid")} '
              f'status={backend.last_worker.get("status_code")}')
    for class_id, score in result.top[:5]:
        print(f'  class {class_id}: score {score:.3f}')

    ok = result.top[0][0] == args.expect_class
    print('OK' if ok else f'FAIL: ожидался класс {args.expect_class}, получен {result.top[0][0]}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()