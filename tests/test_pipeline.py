"""Гоняются БЕЗ VPN: pytest -q из корня репо. Данные синтетические, ассетов нет."""
import numpy as np
import pytest
from PIL import Image

from linq.registry import PROGRAMS, validate_inputs
from linq.vision import load_image, preprocess_resnet
from service.pipeline import Pipeline
from tests.mock_backend import MockBackend


@pytest.fixture()
def image_path(tmp_path):
    path = tmp_path / 'synthetic.jpg'
    Image.new('RGB', (375, 500), (120, 80, 40)).save(path)
    return path


def test_preprocess_shape_and_dtype(image_path):
    tensor = preprocess_resnet(load_image(image_path))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_pipeline_with_mock_backend(image_path):
    result = Pipeline(MockBackend(expected_class=257)).infer_image(image_path)
    assert result.backend == 'mock'
    assert result.top[0][0] == 257
    assert result.out_shapes == {'495': (1, 1000)}


def test_validate_inputs_catches_bad_shape():
    spec = PROGRAMS['resnet50']
    with pytest.raises(ValueError):
        validate_inputs(spec, {'input.1': np.zeros((1, 3, 32, 32), dtype=np.float32)})
    with pytest.raises(KeyError):
        validate_inputs(spec, {'wrong': np.zeros((1, 3, 224, 224), dtype=np.float32)})