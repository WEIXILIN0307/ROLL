from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from roll.distributed.executor.worker import Worker
from roll.pipeline.sft import sft_worker


def make_worker(monkeypatch):
    worker = sft_worker.SFTWorker.__new__(sft_worker.SFTWorker)
    worker.worker_name = "sft_train-0"
    monkeypatch.setattr(Worker, "logger", property(lambda self: Mock()))
    strategy = Mock()
    monkeypatch.setattr(Worker, "initialize", lambda self, pipeline_config: setattr(self, "pipeline_config", pipeline_config))
    monkeypatch.setattr(sft_worker, "create_strategy", Mock(return_value=strategy))
    download = Mock(return_value="/worker-local/checkpoint-2")
    monkeypatch.setattr(sft_worker, "download_model", download, raising=False)
    return worker, strategy, download


def test_sft_worker_restores_model_before_returning_from_initialize(monkeypatch):
    worker, strategy, download = make_worker(monkeypatch)
    cfg = SimpleNamespace(resume_from_checkpoint="remote/checkpoint-2")

    worker.initialize.__wrapped__(worker, pipeline_config=cfg)

    download.assert_called_once_with("remote/checkpoint-2")
    assert strategy.mock_calls == [
        call.initialize(model_provider=sft_worker.default_actor_model_provider),
        call.load_checkpoint(load_dir="/worker-local/checkpoint-2", tag="checkpoint", required=True),
    ]


def test_sft_worker_propagates_checkpoint_load_failure(monkeypatch):
    worker, strategy, _ = make_worker(monkeypatch)
    cfg = SimpleNamespace(resume_from_checkpoint="remote/checkpoint-2")
    strategy.load_checkpoint.side_effect = FileNotFoundError("missing checkpoint shard")

    with pytest.raises(FileNotFoundError, match="missing checkpoint shard"):
        worker.initialize.__wrapped__(worker, pipeline_config=cfg)


def test_sft_worker_does_not_load_checkpoint_for_a_fresh_run(monkeypatch):
    worker, strategy, download = make_worker(monkeypatch)

    worker.initialize.__wrapped__(worker, pipeline_config=SimpleNamespace(resume_from_checkpoint=False))

    strategy.load_checkpoint.assert_not_called()
    download.assert_not_called()


def test_required_fsdp_checkpoint_rejects_worker_state_only(tmp_path):
    from roll.distributed.strategy.fsdp2_strategy import FSDP2TrainStrategy

    strategy = object.__new__(FSDP2TrainStrategy)
    (tmp_path / "pipeline").mkdir()

    with pytest.raises(FileNotFoundError, match="FSDP2 training checkpoint"):
        strategy.load_checkpoint(load_dir=str(tmp_path), required=True)
