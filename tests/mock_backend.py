"""Мок-бэкенд для тестов: без VPN, без TPU, без сети.
Возвращает правдоподобный ответ для resnet50 и проверяет shape входов."""
import time

import numpy as np

from linq.registry import PROGRAMS, validate_inputs


class MockBackend:
    name = 'mock'

    def __init__(self, expected_class: int = 257, delay_s: float = 0.0):
        self.expected_class = expected_class
        self.delay_s = delay_s
        self.calls = []

    def run(self, program: str, tensors: dict) -> dict:
        spec = PROGRAMS[program]
        validate_inputs(spec, tensors)
        if self.delay_s:
            time.sleep(self.delay_s)
        if program != 'resnet50':
            raise NotImplementedError(f'мок знает только resnet50, получен {program!r}')
        scores = np.full((1, 1000), -5.0, dtype=np.float32)
        scores[0, self.expected_class - 1] = 10.0  # классы нумеруются с 1
        self.calls.append((program, {k: v.shape for k, v in tensors.items()}))
        return {spec.outputs[0]: scores}

    def close(self) -> None:
        pass