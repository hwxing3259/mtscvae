"""Run records: the provenance of every fit and every score.

A record is written ONCE, by the process that owns the result, when it finishes
(or by the scheduler when that process was killed and could not).  Logs are not
provenance: two processes can interleave on one log file, never on one record.
The aggregator reads records only.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import socket
import time
from pathlib import Path
from typing import Any

from neuro_dev_model.pfc_scaling_bench import design as D


def record_path(tag: str, kind: str = "fit") -> Path:
    """Return the record path of one fit or score.

    Parameters
    ----------
    tag : str
        Canonical stem from :func:`design.run_tag`.
    kind : {"fit", "score"}
        Which record.

    Returns
    -------
    pathlib.Path
        ``<OUT>/records/<tag>.json`` or ``<OUT>/scores/<tag>.json``.
    """
    sub = "records" if kind == "fit" else "scores"
    return D.out_dir(sub) / f"{tag}.json"


def emb_path(tag: str) -> Path:
    """Return the embedding path of one fit.

    Parameters
    ----------
    tag : str
        Canonical stem.

    Returns
    -------
    pathlib.Path
        ``<OUT>/emb/<tag>.npz``.
    """
    return D.out_dir("emb") / f"{tag}.npz"


def read(tag: str, kind: str = "fit") -> dict[str, Any] | None:
    """Read a record if it exists.

    Parameters
    ----------
    tag : str
        Canonical stem.
    kind : {"fit", "score"}
        Which record.

    Returns
    -------
    dict or None
        The record, or ``None`` when absent.
    """
    p = record_path(tag, kind)
    return json.loads(p.read_text()) if p.exists() else None


def write(tag: str, rec: dict[str, Any], kind: str = "fit") -> Path:
    """Write a record atomically (temp file, then rename).

    Parameters
    ----------
    tag : str
        Canonical stem.
    rec : dict
        Record body.
    kind : {"fit", "score"}
        Which record.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    p = record_path(tag, kind)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, default=str))
    os.replace(tmp, p)
    return p


def cpu_quota() -> float | None:
    """Return the CPU quota in cores (Linux cgroup v2), if any.

    Returns
    -------
    float or None
        e.g. ``23.8``; ``None`` when unlimited or not readable.  ``nproc`` and
        ``os.cpu_count`` report the HOST and are not the budget.
    """
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        return None


def mem_limit_bytes() -> int | None:
    """Return the memory limit (Linux cgroup v2, then v1), if any.

    Returns
    -------
    int or None
        Bytes.  ``free`` reports the host and is not the budget.
    """
    for f in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
            return None if v == "max" else int(v)
        except (OSError, ValueError):
            continue
    return None


def peak_rss_gb() -> float:
    """Return this process's peak resident set size so far.

    Returns
    -------
    float
        GB.  ``ru_maxrss`` is KiB on Linux and bytes on macOS.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2)


def current_rss_gb() -> float:
    """Return this process's current resident set size.

    Returns
    -------
    float
        GB, or NaN if psutil is unavailable.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        return float("nan")


def runtime_facts() -> dict[str, Any]:
    """Facts about where and how a job ran, for every record.

    Returns
    -------
    dict
        Host, quota, threads, lane, the jobs running beside this one at launch,
        GPU name, library versions and the code version.
    """
    import numpy
    facts: dict[str, Any] = {
        "host": socket.gethostname(),
        "cpu_quota_cores": cpu_quota(), "mem_limit_bytes": mem_limit_bytes(),
        "threads": int(os.environ.get("OMP_NUM_THREADS", D.THREADS)),
        "cores": os.environ.get("PFCSB_CORES"),
        "lane": os.environ.get("PFCSB_LANE"),
        "running_beside": json.loads(os.environ.get("PFCSB_BESIDE", "[]")),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(), "numpy": numpy.__version__,
        "unix_time_end": time.time(),
    }
    try:
        import torch
        facts["torch"] = torch.__version__
        facts["gpu"] = (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None)
    except ImportError:
        pass
    cv = D.PROJECT_ROOT / "CODE_VERSION"
    facts["code_version"] = cv.read_text().strip() if cv.exists() else None
    return facts
