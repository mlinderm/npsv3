from omegaconf import OmegaConf

from npsv3.models.runners import _resolve_world_size


class TestResolveWorldSize:
    def test_defaults_to_one(self):
        assert _resolve_world_size(OmegaConf.create({"trainer": {}})) == 1

    def test_explicit_devices_and_nodes(self):
        cfg = OmegaConf.create({"trainer": {"devices": 4, "num_nodes": 2}})
        assert _resolve_world_size(cfg) == 8

    def test_devices_list_counts_entries(self):
        cfg = OmegaConf.create({"trainer": {"devices": [0, 1, 3]}})
        assert _resolve_world_size(cfg) == 3

    def test_auto_devices_falls_back_to_gpu_count(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        cfg = OmegaConf.create({"trainer": {"devices": "auto"}})
        assert _resolve_world_size(cfg) == 1
