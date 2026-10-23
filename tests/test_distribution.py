import json
import sys

import numpy as np
import pytest
from scipy.spatial import QhullError

from bin import analyze_distribution as distribution


COUNTER_NAMES = (
    "n_vertex_draws",
    "n_rejected_vertex_draws",
    "n_configurations_attempted",
    "n_rejected_configurations",
    "n_vertex_generation_failures",
    "n_numerical_evaluation_failures",
    "n_invalid_volume_evaluations",
)


def test_sampling_is_invariant_to_worker_count():
    # One sample beyond a logical batch forces two useful tasks.
    serial = distribution.analyze_distribution(4, 257, seed=1729, n_jobs=1)
    parallel = distribution.analyze_distribution(4, 257, seed=1729, n_jobs=2)

    np.testing.assert_array_equal(serial["volumes"], parallel["volumes"])
    assert serial["effective_workers"] == 1
    assert parallel["effective_workers"] == 2
    assert serial["status"] == parallel["status"] == "complete"
    assert {
        name: serial[name] for name in COUNTER_NAMES
    } == {
        name: parallel[name] for name in COUNTER_NAMES
    }


def test_serial_cli_saves_volume_list_and_sampling_metadata(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "distribution.json"
    plot_path = tmp_path / "distribution.png"
    monkeypatch.setattr(distribution, "plot_distribution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_distribution.py",
            "--vertices",
            "4",
            "--samples",
            "3",
            "--jobs",
            "1",
            "--data",
            str(data_path),
            "--output",
            str(plot_path),
        ],
    )

    distribution.main()

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    assert isinstance(payload["volumes"], list)
    assert len(payload["volumes"]) == 3
    assert payload["metadata"]["status"] == "complete"
    assert payload["metadata"]["effective_workers"] == 1
    for name in COUNTER_NAMES:
        assert isinstance(payload["metadata"][name], int)


@pytest.mark.parametrize("n_vertices", [0, 3])
def test_sampling_requires_at_least_four_vertices(n_vertices):
    with pytest.raises(ValueError, match="at least 4"):
        distribution.analyze_distribution(n_vertices, 1, n_jobs=1)


def test_bootstrap_is_local_reproducible_and_supports_fewer_than_ten_draws():
    volumes = np.linspace(0.25, 2.0, 20)

    first = distribution.fit_distribution(
        volumes,
        dist_name="norm",
        n_bootstrap=7,
        seed=8675309,
    )
    second = distribution.fit_distribution(
        volumes,
        dist_name="norm",
        n_bootstrap=7,
        seed=8675309,
    )

    assert first["bootstrap"] == second["bootstrap"]
    assert first["confidence_intervals"] == second["confidence_intervals"]
    with pytest.raises(ValueError, match="positive"):
        distribution.fit_distribution(volumes, dist_name="norm", n_bootstrap=0)


def test_fit_distribution_uses_a_bound_cdf_callable(monkeypatch):
    original_kstest = distribution.stats.kstest
    observed = {}

    def checking_kstest(data, cdf, *args, **kwargs):
        observed["cdf"] = cdf
        return original_kstest(data, cdf, *args, **kwargs)

    monkeypatch.setattr(distribution.stats, "kstest", checking_kstest)

    distribution.fit_distribution(
        np.linspace(0.25, 2.0, 20),
        dist_name="norm",
        n_bootstrap=1,
        seed=123,
    )

    assert callable(observed["cdf"])


def test_fit_distribution_selects_new_anderson_method(monkeypatch):
    original_anderson = distribution.stats.anderson
    observed = {}

    def compatible_anderson(data, dist="norm", *, method=None):
        observed["method"] = method
        kwargs = {}
        if "method" in distribution.inspect.signature(original_anderson).parameters:
            kwargs["method"] = method
        return original_anderson(data, dist=dist, **kwargs)

    monkeypatch.setattr(distribution.stats, "anderson", compatible_anderson)

    distribution.fit_distribution(
        np.linspace(0.25, 2.0, 20),
        dist_name="norm",
        n_bootstrap=1,
        seed=123,
    )

    assert observed["method"] == "interpolate"


def test_unexpected_sampling_error_propagates(monkeypatch):
    def fail_unexpectedly(*args, **kwargs):
        raise RuntimeError("programming error")

    monkeypatch.setattr(
        distribution,
        "ideal_poly_volume_via_delaunay",
        fail_unexpectedly,
    )

    with pytest.raises(RuntimeError, match="programming error"):
        distribution.analyze_distribution(4, 1, seed=1, n_jobs=1)


def test_known_numerical_failure_is_retried_and_counted(monkeypatch):
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise QhullError("degenerate sample")
        return 1.25

    monkeypatch.setattr(
        distribution,
        "ideal_poly_volume_via_delaunay",
        fail_once,
    )

    result = distribution.analyze_distribution(4, 1, seed=1, n_jobs=1)

    np.testing.assert_array_equal(result["volumes"], np.array([1.25]))
    assert result["n_numerical_evaluation_failures"] == 1
    assert result["n_rejected_configurations"] == 1
    assert result["n_configurations_attempted"] == 2


def test_attempt_limit_reports_underfill(monkeypatch):
    monkeypatch.setattr(
        distribution,
        "ideal_poly_volume_via_delaunay",
        lambda *args, **kwargs: np.nan,
    )
    monkeypatch.setattr(
        distribution,
        "MAX_CONFIGURATION_ATTEMPTS_PER_SAMPLE",
        2,
    )

    result = distribution.analyze_distribution(4, 2, seed=1, n_jobs=1)

    assert isinstance(result["volumes"], np.ndarray)
    assert result["volumes"].size == 0
    assert result["status"] == "attempt_limit_reached"
    assert result["complete"] is False
    assert result["n_missing"] == 2
    assert result["n_rejected_configurations"] == 4


@pytest.mark.parametrize("n_jobs", [0, -1])
def test_sampling_rejects_nonpositive_worker_counts(n_jobs):
    with pytest.raises(ValueError, match="n_jobs must be positive"):
        distribution.analyze_distribution(4, 1, n_jobs=n_jobs)


def test_cli_preserves_empty_underfill_diagnostics_without_plotting(
    monkeypatch,
    tmp_path,
):
    data_path = tmp_path / "underfilled.json"
    monkeypatch.setattr(
        distribution,
        "analyze_distribution",
        lambda *args, **kwargs: {
            "volumes": np.array([], dtype=float),
            "n_samples_requested": 1,
            "n_valid": 0,
            "n_missing": 1,
            "complete": False,
            "status": "attempt_limit_reached",
            "effective_workers": 1,
            "logical_batch_size": distribution.LOGICAL_BATCH_SIZE,
            "n_logical_batches": 1,
            "max_vertex_draw_attempts": distribution.MAX_VERTEX_DRAW_ATTEMPTS,
            "max_configuration_attempts_per_sample": (
                distribution.MAX_CONFIGURATION_ATTEMPTS_PER_SAMPLE
            ),
            **{name: 0 for name in COUNTER_NAMES},
            **{
                name: float("nan")
                for name in ("mean", "median", "std", "min", "max", "q25", "q75")
            },
        },
    )
    monkeypatch.setattr(
        distribution,
        "plot_distribution",
        lambda *args, **kwargs: pytest.fail("empty output must not be plotted"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_distribution.py",
            "--vertices",
            "4",
            "--samples",
            "1",
            "--data",
            str(data_path),
            "--output",
            str(tmp_path / "unused.png"),
        ],
    )

    distribution.main()

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["complete"] is False
    assert payload["metadata"]["n_missing"] == 1
    assert payload["volumes"] == []
    assert payload["statistics"] == {
        name: None
        for name in ("mean", "median", "std", "min", "max", "q25", "q75")
    }
