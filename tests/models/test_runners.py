import os

import pytest

from npsv3.models.runners import _resolve_resume_checkpoint


def _touch(path, mtime):
    path.write_text("")
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
