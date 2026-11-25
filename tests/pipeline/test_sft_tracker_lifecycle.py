from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from roll.pipeline import base_pipeline
from roll.pipeline.sft import sft_pipeline


def make_config():
    return SimpleNamespace(
        seed=1,
        num_nodes=1,
        num_gpus_per_node=0,
        checkpoint_config={},
        to_dict=dict,
        track_with="rl_insight",
        tracker_kwargs={},
        transfer_backend=None,
        resume_from_checkpoint=False,
        auto_resume=False,
        sft_train=SimpleNamespace(model_args=None, training_args=SimpleNamespace(num_train_epochs=1)),
    )


def driver_without_batches(tracker):
    driver = sft_pipeline.SFTPipeline.__new__(sft_pipeline.SFTPipeline)
    driver.tracker = tracker
    driver.pipeline_config = make_config()
    driver.dataloader = []
    driver.resume_futures = [Mock(spec=Future)]
    return driver


def test_sft_drains_replay_before_finishing_without_batches():
    tracker = Mock()
    driver = driver_without_batches(tracker)
    replay = driver.resume_futures[0]
    tracker.attach_mock(replay.result, "wait_for_replay")

    driver.run()

    assert [call[0] for call in tracker.mock_calls] == ["wait_for_replay", "finish"]
    assert not driver.resume_futures


def test_sft_tracker_cleanup_does_not_mask_training_failure(monkeypatch):
    tracker = Mock()
    tracker.finish.side_effect = RuntimeError("monitor cleanup failed")
    driver = driver_without_batches(tracker)
    error = RuntimeError("training failed")
    monkeypatch.setattr(sft_pipeline, "MetricsManager", Mock(side_effect=error))

    with pytest.raises(RuntimeError) as raised:
        driver.run()

    assert raised.value is error
    tracker.finish.assert_called_once_with()


def test_sft_initialization_failure_finishes_created_tracker(monkeypatch):
    tracker = Mock()
    monkeypatch.setattr(base_pipeline, "ResourceManager", Mock())
    monkeypatch.setattr(base_pipeline, "create_tracker", Mock(return_value=tracker))
    monkeypatch.setattr(base_pipeline, "init_transfer_backend", Mock())
    monkeypatch.setattr(sft_pipeline, "default_tokenizer_provider", Mock(side_effect=RuntimeError("tokenizer failed")))

    with pytest.raises(RuntimeError, match="tokenizer failed"):
        sft_pipeline.SFTPipeline(make_config())

    tracker.finish.assert_called_once_with()


def test_unsupported_pipeline_rejects_rl_insight_before_creating_resources(monkeypatch):
    create_resource_manager = Mock()
    create_tracker = Mock()
    monkeypatch.setattr(base_pipeline, "ResourceManager", create_resource_manager)
    monkeypatch.setattr(base_pipeline, "create_tracker", create_tracker)

    with pytest.raises(ValueError, match="track_with=rl_insight.*BasePipeline"):
        base_pipeline.BasePipeline(make_config())

    create_resource_manager.assert_not_called()
    create_tracker.assert_not_called()
