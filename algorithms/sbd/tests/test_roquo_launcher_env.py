"""ROQUO launcher helpers (algorithms/sbd/roquo/bs_common.py, run_task.py): Prefect storage.

Regression: the launchers set PREFECT_HOME / PREFECT_LOCAL_STORAGE_PATH only after the manifest
was loaded, i.e. after Prefect was imported, so the change was ignored and pools were written to
~/.prefect/storage (where find_saved_pools never looks). Each case runs in a fresh interpreter
because Prefect caches its settings globally.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROQUO = Path(__file__).resolve().parents[1] / "roquo"

_SCRIPT = """
import os, sys
sys.path.insert(0, {roquo!r})
import qcsc_workflow_utility.active_space  # noqa: F401  (pulls in Prefect, as load_spec does)
import prefect  # noqa: F401
import bs_common as C
from pathlib import Path
C.isolate_prefect(Path({home!r}), storage_only={storage_only})
from prefect.settings import get_current_settings
from sbd.data_io import _local_storage_dir
s = get_current_settings()
print("HOME", s.home)
print("STORAGE", C.local_storage_dir())
print("DATAIO", _local_storage_dir())
"""


def _run(tmp_path, storage_only):
    home = tmp_path / "run"
    env = {**os.environ, "PREFECT_HOME": str(tmp_path / "outer")}
    env.pop("PREFECT_LOCAL_STORAGE_PATH", None)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            _SCRIPT.format(roquo=str(_ROQUO), home=str(home), storage_only=storage_only),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    vals = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
    return home, vals


@pytest.mark.parametrize("storage_only", [False, True])
def test_isolate_prefect_takes_effect_after_prefect_import(tmp_path, storage_only):
    home, vals = _run(tmp_path, storage_only)
    assert Path(vals["STORAGE"]) == home / "storage"
    assert Path(vals["DATAIO"]) == home / "storage"  # where save_ndarray writes the pool
    expected_home = (tmp_path / "outer") if storage_only else home
    assert Path(vals["HOME"]) == expected_home  # storage_only keeps a Cloud profile's home
    assert (home / "storage").is_dir()


def test_launcher_uses_run_flow_and_per_run_trace():
    task = (_ROQUO / "run_task.py").read_text()
    assert "C.run_flow(params" in task
    assert "C.read_recovery_trace(flow_run_id)" in task
    # a sampling run persists its pool under the task's samples/ (storage_only: Cloud profile)
    assert "C.isolate_prefect(store, storage_only=True)" in task
