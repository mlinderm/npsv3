import types

import torch

from npsv3.models.paired import InOutEncoder


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=16, use_cache=False)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return types.SimpleNamespace(last_hidden_state=torch.randn(batch_size, 1, 16))


class DummyProcessor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [1.0, 1.0, 1.0]


def test_inout_encoder_smoke(monkeypatch):
    monkeypatch.setattr("transformers.AutoImageProcessor.from_pretrained", lambda *args, **kwargs: DummyProcessor())
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *args, **kwargs: DummyBackbone())

    encoder = InOutEncoder("dummy-model", chunk_size=2, num_channels=7, projection_size=8)
    images = torch.randn(4, 7, 16, 16)

    embeddings = encoder(images)

    assert embeddings.shape == (4, 8)
