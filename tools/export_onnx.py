"""Экспорт готовых моделей в ONNX (статический batch=1) для последующей компиляции в .tpu.
Запуск: python -m tools.export_onnx --model yolo_detect
Нужно: pip install -r tools/requirements.txt
Артефакты (.onnx) в гит не коммитятся (gitignore: artifacts/)."""
import argparse
from pathlib import Path

MODELS = {
    # имя -> (веса ultralytics, размер входа)
    'yolo_detect': ('yolov8n.pt', 640),
    'yolo_pose': ('yolov8n-pose.pt', 640),
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Экспорт модели в ONNX')
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts'))
    args = parser.parse_args()

    from ultralytics import YOLO  # тяжёлый импорт, нужен только здесь

    source, imgsz = MODELS[args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(source)
    path = model.export(format='onnx', opset=13, simplify=True, imgsz=imgsz, batch=1)
    dest = args.out_dir / f'{args.model}.onnx'
    Path(path).replace(dest)
    print(f'ONNX: {dest} (batch=1, imgsz={imgsz}, opset=13, simplify=True)')


if __name__ == '__main__':
    main()