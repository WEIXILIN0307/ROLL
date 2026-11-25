import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from roll.pipeline import base_pipeline
from roll.pipeline.rlvr import rlvr_pipeline, rlvr_vlm_pipeline
from roll.pipeline.rlvr.rlvr_rollout_pipeline import RLVRRolloutPipeline
from roll.pipeline.rlvr.rlvr_vlm_rollout_pipeline import RLVRVLMPipeline as VisionRolloutPipeline


@pytest.fixture(
    params=[
        (rlvr_pipeline, rlvr_pipeline.RLVRPipeline, "default_tokenizer_provider"),
        (rlvr_vlm_pipeline, rlvr_vlm_pipeline.RLVRVLMPipeline, "default_processor_provider"),
    ],
    ids=["text", "vision"],
)
def driver_entry(request):
    return request.param


def driver_without_training(driver_type, tracker, *, backend="rl_insight"):
    driver = driver_type.__new__(driver_type)
    driver.tracker = tracker
    driver.pipeline_config = SimpleNamespace(track_with=backend, async_pipeline=False, max_steps=0)
    driver.resume_futures = []
    driver.generate_schedulers = {"train": Mock()}
    driver.val_dataset = None
    return driver


def test_run_joins_resume_even_when_no_checkpoint(monkeypatch, driver_entry):
    module, driver_type, _ = driver_entry
    tracker = Mock()
    driver = driver_without_training(driver_type, tracker)
    shutdown_called = threading.Event()
    allow_replay_to_finish = threading.Event()
    order = []

    def shutdown(refs):
        order.append("scheduler shutdown")
        shutdown_called.set()

    def replay():
        assert allow_replay_to_finish.wait(timeout=5)
        order.append("replay")
        tracker.log({"loss": 1}, step=2)

    tracker.finish.side_effect = lambda: order.append("tracker finish")
    monkeypatch.setattr(module.ray, "get", shutdown)
    with ThreadPoolExecutor(max_workers=2) as executor:
        driver.resume_futures = [executor.submit(replay)]
        run = executor.submit(driver.run)
        try:
            assert shutdown_called.wait(timeout=5)
        finally:
            allow_replay_to_finish.set()
        run.result(timeout=5)

    assert order == ["scheduler shutdown", "replay", "tracker finish"]
    tracker.finish.assert_called_once()


@pytest.mark.parametrize("phase", ["training", "scheduler"])
def test_run_preserves_primary_failure(monkeypatch, driver_entry, phase):
    module, driver_type, _ = driver_entry
    tracker = Mock()
    tracker.finish.side_effect = RuntimeError("monitor cleanup")
    report = Mock()
    monkeypatch.setattr(module.logger, "exception", report)
    driver = driver_without_training(driver_type, tracker)
    error = KeyboardInterrupt() if phase == "training" else RuntimeError("scheduler cleanup")
    if phase == "training":
        monkeypatch.setattr(module, "MetricsManager", Mock(side_effect=error))
    else:
        monkeypatch.setattr(module.ray, "get", Mock(side_effect=error))
    replay = Future()
    replay.set_exception(RuntimeError("broken history"))
    driver.resume_futures = [replay]

    with pytest.raises(type(error)) as raised:
        driver.run()
    assert raised.value is error
    tracker.finish.assert_called_once()
    assert report.call_count == 2


@pytest.mark.parametrize("phase", ["base", "rlvr"])
def test_constructor_failure_closes_created_tracker(monkeypatch, driver_entry, phase):
    module, driver_type, provider = driver_entry
    tracker = Mock()
    error = RuntimeError("initialization failed")
    monkeypatch.setattr(base_pipeline, "ResourceManager", Mock())
    monkeypatch.setattr(base_pipeline, "create_tracker", Mock(return_value=tracker))
    monkeypatch.setattr(base_pipeline, "init_transfer_backend", Mock(side_effect=error if phase == "base" else None))
    monkeypatch.setattr(module, provider, Mock(side_effect=error))
    config = SimpleNamespace(
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
        enable_reference=False,
        actor_train=SimpleNamespace(model_args=None),
    )

    with pytest.raises(RuntimeError) as raised:
        driver_type(config)
    assert raised.value is error
    tracker.finish.assert_called_once()


def test_rlvr_leaves_other_tracker_lifecycles_unchanged(monkeypatch, driver_entry):
    module, driver_type, _ = driver_entry
    tracker = Mock()
    driver = driver_without_training(driver_type, tracker, backend="tensorboard")
    monkeypatch.setattr(module.ray, "get", Mock())
    driver.run()
    tracker.finish.assert_not_called()


@pytest.mark.parametrize("driver_cls", [base_pipeline.BasePipeline, RLVRRolloutPipeline, VisionRolloutPipeline])
def test_unsupported_driver_rejects_rl_insight_before_allocating_resources(monkeypatch, driver_cls):
    config = SimpleNamespace(track_with="rl_insight", seed=1, num_nodes=1, num_gpus_per_node=0)
    resource_manager = Mock(side_effect=AssertionError("unsupported driver reached resource allocation"))
    create_tracker = Mock()
    monkeypatch.setattr(base_pipeline, "ResourceManager", resource_manager)
    monkeypatch.setattr(base_pipeline, "create_tracker", create_tracker)

    with pytest.raises(ValueError, match=f"track_with=rl_insight.*{driver_cls.__name__}"):
        driver_cls(config)

    resource_manager.assert_not_called()
    create_tracker.assert_not_called()


def test_cancelled_replay_and_failed_flush_do_not_fail_a_successful_run(monkeypatch, driver_entry):
    module, driver_type, _ = driver_entry
    tracker = Mock()
    tracker.finish.side_effect = RuntimeError("monitor cleanup")
    report = Mock()
    monkeypatch.setattr(module.logger, "exception", report)
    monkeypatch.setattr(module.ray, "get", Mock())
    driver = driver_without_training(driver_type, tracker)
    replay = Future()
    assert replay.cancel()
    driver.resume_futures = [replay]

    driver.run()

    tracker.finish.assert_called_once_with()
    assert not driver.resume_futures
    assert [call.args[0] for call in report.call_args_list] == [
        "Failed to replay checkpoint metrics",
        "Failed to finish the RL-Insight tracker",
    ]
