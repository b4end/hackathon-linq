"""Контракт бэкенда: словарь тензоров внутрь -> словарь наружу.
Пайплайну всё равно, кто считает: облачный TPU или мок в тестах."""
from typing import Dict, Protocol

import numpy as np

Tensors = Dict[str, np.ndarray]


class InferenceBackend(Protocol):
    name: str

    def run(self, program: str, tensors: Tensors) -> Tensors:
        ...

    def close(self) -> None:
        ...