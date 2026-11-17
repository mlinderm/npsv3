import logging
import os

import pytest
import torch

from npsv3.models.runners import _resolve_resume_checkpoint


def _touch(path, mtime, *, weight=1.0, scale=None):
    """Write a minimal Lightning-like checkpoint, with a `weight` (e.g., NaN) and optional GradScaler `scale`"""
    checkpoint = {
        "state_dict": {"layer.weight": torch.full((2, 2), weight), "layer.num_batches_tracked": torch.tensor(3)},
        "optimizer_states": [{"state": {0: {"exp_avg": torch.zeros(2, 2)}}, "param_groups": [{"lr": 1e-3}]}],
    }
    if scale is not None:
        checkpoint["MixedPrecision"] = {"scale": scale, "growth_factor": 2.0}
    torch.save(checkpoint, path)
    os.utime(path, (mtime, mtime))
    return str(path)


class TestResolveResumeCheckpoint:
    def test_none_defers_to_lightning(self, tmp_path):
        _touch(tmp_path / "hpc_ckpt_1.ckpt", 100)
        assert _resolve_resume_checkpoint(None, str(tmp_path)) is None

    def test_last_selects_most_recent_of_any_checkpoint(self, tmp_path):
        _touch(tmp_path / "epoch=0-step=10.ckpt", 100)
        hpc = _touch(tmp_path / "hpc_ckpt_1.ckpt", 200)
        assert _resolve_resume_checkpoint("last", str(tmp_path)) == hpc

        # A stale requeue checkpoint is superseded by a more recent epoch checkpoint
        epoch = _touch(tmp_path / "epoch=1-step=20.ckpt", 300)
        assert _resolve_resume_checkpoint("last", str(tmp_path)) == epoch

    def test_last_ignores_symlinks(self, tmp_path):
        epoch = _touch(tmp_path / "epoch=0-step=10.ckpt", 100)
        os.symlink(epoch, tmp_path / "model.ckpt")
        assert _resolve_resume_checkpoint("last", str(tmp_path)) == epoch

    def test_last_without_checkpoints_trains_from_scratch(self, tmp_path):
        assert _resolve_resume_checkpoint("last", str(tmp_path)) is None

    def test_path_superseded_by_more_recent_checkpoint(self, tmp_path):
        """A requeued job re-runs the same `resume_from`, but must continue from its own, more recent, checkpoint"""
        other = tmp_path / "other"
        other.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        original = _touch(other / "epoch=3-step=30.ckpt", 100)
        assert _resolve_resume_checkpoint(original, str(output)) == original

        hpc = _touch(output / "hpc_ckpt_1.ckpt", 200)
        assert _resolve_resume_checkpoint(original, str(output)) == hpc

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_resume_checkpoint(str(tmp_path / "missing.ckpt"), str(tmp_path))


class TestResolveResumeCheckpointSkipsUnusable:
    def test_skips_zero_grad_scale(self, tmp_path, caplog):
        good = _touch(tmp_path / "epoch=0-step=10.ckpt", 100, scale=65536.0)
        bad = _touch(tmp_path / "epoch=1-step=20.ckpt", 200, scale=0.0)
        with caplog.at_level(logging.WARNING):
            assert _resolve_resume_checkpoint("last", str(tmp_path)) == good
        assert bad in caplog.text and "scale" in caplog.text

    def test_skips_nonfinite_weights(self, tmp_path, caplog):
        good = _touch(tmp_path / "epoch=0-step=10.ckpt", 100)
        _touch(tmp_path / "epoch=1-step=20.ckpt", 200, weight=float("nan"))
        _touch(tmp_path / "hpc_ckpt_1.ckpt", 300, weight=float("inf"))
        with caplog.at_level(logging.WARNING):
            assert _resolve_resume_checkpoint("last", str(tmp_path)) == good
        assert caplog.text.count("Skipping checkpoint") == 2

    def test_skips_unreadable(self, tmp_path):
        good = _touch(tmp_path / "epoch=0-step=10.ckpt", 100)
        truncated = tmp_path / "hpc_ckpt_1.ckpt"
        truncated.write_bytes(b"not a checkpoint")
        os.utime(truncated, (200, 200))
        assert _resolve_resume_checkpoint("last", str(tmp_path)) == good

    def test_stops_at_newest_usable(self, tmp_path, monkeypatch):
        """Candidates are checked newest first, and older ones aren't loaded once a usable one is found"""
        _touch(tmp_path / "epoch=0-step=10.ckpt", 100)
        newest = _touch(tmp_path / "epoch=1-step=20.ckpt", 200)
        loaded = []
        original_load = torch.load

        def load(path, *args, **kwargs):
            loaded.append(str(path))
            return original_load(path, *args, **kwargs)

        monkeypatch.setattr(torch, "load", load)
        assert _resolve_resume_checkpoint("last", str(tmp_path)) == newest
        assert loaded == [newest]

    def test_skips_unusable_resume_from_path(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        output = tmp_path / "output"
        output.mkdir()
        bad = _touch(other / "epoch=3-step=30.ckpt", 200, weight=float("nan"))
        good = _touch(output / "epoch=2-step=20.ckpt", 100)
        assert _resolve_resume_checkpoint(bad, str(output)) == good

    def test_raises_if_none_usable(self, tmp_path):
        _touch(tmp_path / "epoch=0-step=10.ckpt", 100, scale=0.0)
        _touch(tmp_path / "epoch=1-step=20.ckpt", 200, weight=float("nan"))
        with pytest.raises(RuntimeError, match="None of the checkpoints"):
            _resolve_resume_checkpoint("last", str(tmp_path))
