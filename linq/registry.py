"""Реестр программ для TPU: только метаданные (файлы живут вне гита)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgramSpec:
    name: str
    file: str            # имя файла программы в /auto/cache/<логин>/
    inputs: dict         # имя тензора -> shape (tuple)
    outputs: tuple       # имена выходных тензоров
    image_hw: tuple      # (H, W) входа модели
    preprocess: str      # ключ linq.vision.PREPROCESSORS
    postprocess: str     # ключ linq.vision.POSTPROCESSORS


PROGRAMS = {
    'resnet50': ProgramSpec(
        name='resnet50',
        file='torch_resnet50_b1_o5_h1m.tpu',
        inputs={'input.1': (1, 3, 224, 224)},
        outputs=('495',),
        image_hw=(224, 224),
        preprocess='resnet',
        postprocess='resnet',
    ),
    # 'yolo_detect': ProgramSpec(...),  # появится, когда скомпилируем свои программы
}


def validate_inputs(spec: ProgramSpec, tensors: dict) -> None:
    """Проверка тензоров ДО отправки: ловим баги препроцессинга без сети."""
    for name, shape in spec.inputs.items():
        if name not in tensors:
            raise KeyError(f'нет входного тензора {name!r} для программы {spec.name!r}')
        got = tuple(tensors[name].shape)
        if got != tuple(shape):
            raise ValueError(f'{spec.name}:{name} shape {got} != ожидаемый {tuple(shape)}')