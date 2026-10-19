"""The background pipeline job can be asked to stop between tiles."""

from __future__ import annotations

import threading
import time

from src.main_pipeline.job import claim_pipeline_result, pipeline_status, start_pipeline, stop_pipeline
from src.main_pipeline.runner import PipelineStopped


def test_stop_pipeline_finishes_without_a_result(monkeypatch) -> None:
    started = threading.Event()

    def fake_run(_config, progress_cb=None, *, write_outputs=True, should_stop=None):
        del progress_cb, write_outputs
        started.set()
        while should_stop is None or not should_stop():
            time.sleep(0.01)
        raise PipelineStopped("Pipeline stopped.")

    monkeypatch.setattr("src.main_pipeline.job.run_pipeline", fake_run)
    start_pipeline({"input_dir": "tiles"})
    assert started.wait(2)
    stop_pipeline()
    assert pipeline_status()["stopping"]

    deadline = time.time() + 2
    while pipeline_status()["running"] and time.time() < deadline:
        time.sleep(0.01)

    assert pipeline_status()["running"] is False
    result = claim_pipeline_result()
    assert result is not None
    assert result["stopped"] is True
    assert result["table"] is None
    assert claim_pipeline_result() is None
