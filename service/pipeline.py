"""Пайплайн уровня приложения: картинка -> preprocess -> backend -> postprocess.
Сюда позже встанут кадры из видео/камер, трекер и правила нарушений."""
import time
from dataclasses import dataclass

from linq.registry import PROGRAMS, validate_inputs
from linq.vision import POSTPROCESSORS, PREPROCESSORS, load_image


@dataclass
class ImageResult:
    source: str
    program: str
    backend: str
    top: list            # [(class_id, score), ...]
    latency_s: float
    out_shapes: dict


class Pipeline:
    def __init__(self, backend):
        self.backend = backend

    def infer_image(self, path, program: str = 'resnet50') -> ImageResult:
        spec = PROGRAMS[program]
        image = load_image(path)
        preprocess = PREPROCESSORS[spec.preprocess]
        tensors = {name: preprocess(image, spec.image_hw) for name in spec.inputs}
        validate_inputs(spec, tensors)

        started = time.monotonic()
        out = self.backend.run(program, tensors)
        latency = time.monotonic() - started

        postprocess = POSTPROCESSORS[spec.postprocess]
        top = postprocess(out[spec.outputs[0]])
        return ImageResult(
            source=str(path),
            program=program,
            backend=self.backend.name,
            top=top,
            latency_s=latency,
            out_shapes={k: tuple(v.shape) for k, v in out.items()},
        )