"""Компиляция ONNX -> Программа для TPU (.tpu) через SDK tpu_framework.
SDK ставится вручную по инструкции (см. README, «SDK и компиляция»). Запуск на машине с SDK:
    python -m tools.compile_tpu --onnx artifacts/yolo_detect.onnx --name yolo_detect \
        --input-nodes '{"images": [1, 3, 640, 640]}' --calib-dir data/calib --preprocess yolo
Готовый .tpu в гит не коммитится; заливается в /auto/cache/<логин>/ вручную.
Основано на их примере compile.py."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from PIL import Image

try:
    from tpu_framework import (
        O1, O5, TPU_H1M, Network, QuantizedModel, RegularModel, TpuProgram, compiler,
        freeze_shapes, preliminary_model_conditioning,
    )
except ImportError:
    sys.exit('tpu_framework не установлен. См. README, раздел «SDK и компиляция».')

from linq.vision import PREPROCESSORS

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def remove_initialized_inputs(model_proto) -> None:
    """Убрать из inputs те, что лежат в initializer (как в их compile.py)."""
    init_names = {i.name for i in model_proto.graph.initializer}
    for inp in list(model_proto.graph.input):
        if inp.name in init_names:
            model_proto.graph.input.remove(inp)


def build_calibration(preprocess_name: str, calib_dir: Path, shape, limit: int) -> np.ndarray:
    """Калибровочный тензор из РЕАЛЬНЫХ препроцессированных кадров (в их примере — random).
    Точный ожидаемый формат calibration_data сверить с tpu_framework.pdf, раздел
    "TPU End-To-End Test"; здесь кладём пак (N, C, H, W)."""
    preprocess = PREPROCESSORS[preprocess_name]
    files = [p for p in sorted(calib_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        sys.exit(f'в {calib_dir} нет картинок для калибровки')
    arrays = [preprocess(Image.open(p).convert('RGB'), tuple(shape[2:])) for p in files[:limit]]
    return np.concatenate(arrays, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description='Компиляция ONNX в .tpu')
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--input-nodes', required=True,
                        help='JSON со статическими shape входов, напр. {"images": [1,3,640,640]}')
    parser.add_argument('--calib-dir', default=None, help='папка с кадрами для калибровки')
    parser.add_argument('--preprocess', choices=sorted(PREPROCESSORS), default=None)
    parser.add_argument('--calib-max', type=int, default=64)
    parser.add_argument('--preset', choices=('O5', 'O1'), default='O5')
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    if bool(args.calib_dir) != bool(args.preprocess):
        parser.error('--calib-dir и --preprocess указываются вместе')

    save_dir = args.out_dir / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    input_nodes = {k: list(v) for k, v in json.loads(args.input_nodes).items()}

    print('1/3 optimization')
    model = onnx.load(args.onnx)
    remove_initialized_inputs(model)
    graph_shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                    for i in model.graph.input}
    for name, shape in graph_shapes.items():
        if 0 in shape and name not in input_nodes:
            sys.exit(f'вход {name} динамический и не задан в --input-nodes')
    freeze_shapes(model, input_nodes)
    model = preliminary_model_conditioning(model)
    opt_path = save_dir / f'{args.name}_opt.onnx'
    onnx.save(model, str(opt_path))

    print('2/3 quantization')
    model = onnx.load(str(opt_path))
    input_nodes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                   for i in model.graph.input}
    shapes = list(input_nodes.values())
    batch = shapes[0][0]
    if args.calib_dir:
        if len(input_nodes) != 1:
            sys.exit('калибровка реальными данными для нескольких входов ещё не собрана, '
                     'уберите --calib-dir')
        stack = build_calibration(args.preprocess, Path(args.calib_dir), shapes[0], args.calib_max)
        calibration_dict = {list(input_nodes)[0]: stack}
    else:
        calibration_dict = {n: np.random.rand(*s).astype(np.float32)
                            for n, s in input_nodes.items()}
    regular = RegularModel(model=model)
    thresholds = regular.calibrate_torch(calibration_data=calibration_dict)
    RegularModel.dump_thresholds(str(save_dir / 'thresholds.json'), thresholds)
    quant = regular.quantize(thresholds)
    quant.save(file_dir=str(save_dir), file_name=f'{args.name}.qm.onnx')
    quant.as_torch_graph(batch_size=batch).export_to_pt2(str(save_dir / f'{args.name}.pt2'))

    print('3/3 compilation')
    preset = O5 if args.preset == 'O5' else O1
    program_name = f'{args.name}_b{batch}{args.preset.lower()}_h1m.tpu'
    quantized = QuantizedModel.load(str(save_dir / f'{args.name}.qm.onnx'))
    network, _ = Network.from_quantized_model(quantized)
    try:
        tlm = compiler.compile_(hardware_parameters=TPU_H1M, network=network, parameters=preset)
    except Exception as err:
        if preset is O5:
            print(f'O5 не собрался ({err}), пробуем O1')
            tlm = compiler.compile_(hardware_parameters=TPU_H1M, network=network, parameters=O1)
            program_name = f'{args.name}_b{batch}o1_h1m.tpu'
        else:
            raise
    executable, tensor_descriptions = tlm
    TpuProgram.from_executable(executable, tensor_descriptions).to_file(str(save_dir / program_name))
    print(f'OK: {save_dir / program_name}')


if __name__ == '__main__':
    main()