"""Performance tests: ensure Feature class completes in reasonable time.

Marked as slow -- run with: pytest tests/test_performance.py -v
"""
import time
import pytest

import ucbfaculty.new_feature as nf
from ucbfaculty.rawdata import clear_dataframe_cache


def _skip_on_data_error(func):
    """Decorator to skip tests when data files are unavailable."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"Data unavailable: {exc}")
        except Exception as exc:
            if "decrypt" in str(exc).lower():
                pytest.skip(f"Cannot decrypt: {exc}")
            raise
    return wrapper


def _time_it(func, iterations=3):
    """Run func multiple times and return median elapsed time."""
    times = []
    for _ in range(iterations):
        clear_dataframe_cache()
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


@pytest.mark.slow
@_skip_on_data_error
def test_perf_simple_feature():
    """Simple Email-indexed feature (FullName)."""
    def run():
        feat = nf.Feature("FullName")
        feat["latest"]

    elapsed = _time_it(run)
    assert elapsed < 5.0, f"FullName took {elapsed:.3f}s (limit 5s)"


@pytest.mark.slow
@_skip_on_data_error
def test_perf_fullname_mapped_feature():
    """Feature requiring FullName->Email mapping (Division)."""
    def run():
        feat = nf.Feature("Division")
        feat["latest"]

    elapsed = _time_it(run)
    assert elapsed < 5.0, f"Division took {elapsed:.3f}s (limit 5s)"


@pytest.mark.slow
@_skip_on_data_error
def test_perf_history():
    """Multi-year history (Appointment)."""
    def run():
        feat = nf.Feature("Appointment")
        feat["history"]

    elapsed = _time_it(run, iterations=2)
    assert elapsed < 10.0, f"Appointment history took {elapsed:.3f}s (limit 10s)"
