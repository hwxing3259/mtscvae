"""Refit the paper's seeded runs from a run list, one at a time, and check them.
Example: ``python reproducibility/run_all.py reproducibility/pfc_full.yaml --out runs/``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import compare as cmp  # noqa: E402

log = logging.getLogger("run_all")


def fit_argv(spec: Dict[str, Any], run: Dict[str, Any], root: Path, out: Path) -> List[str]:
    """Command line of ``mtvae.fit`` for one run.

    Parameters
    ----------
    spec : dict
        The parsed run list.
    run : dict
        One entry of ``runs`` (``seed``, and ``dataset`` for four-dataset runs).
    root : pathlib.Path
        Figshare data root.
    out : pathlib.Path
        Output root; the run writes ``out/<folder>``.

    Returns
    -------
    list of str
    """
    folder = spec["folder"].format(**run)
    argv = [sys.executable, "-m", "mtvae.fit", "--config", str(REPO / spec["config"]),
            "--seed", str(run["seed"]), "--data-root", str(root),
            "--arm", folder, "--out-root", str(out)]
    if "dataset" in run:
        argv += ["--dataset", run["dataset"]]
    return argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every listed fit, sequentially, and compare each with its release.

    Parameters
    ----------
    argv : sequence of str, optional

    Returns
    -------
    int
        0 if every run reproduced its released fit, 3 otherwise, 1 if a fit failed.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_list", type=Path)
    ap.add_argument("--data-root", type=Path, default=os.environ.get("MTVAE_DATA"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help="keep runs whose seed, dataset or folder is listed, e.g. "
                         "44, HC or fourds/HC/s0")
    ap.add_argument("--no-compare", action="store_true",
                    help="fit only (e.g. when producing the released fits)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.data_root is None:
        raise SystemExit("pass --data-root or set MTVAE_DATA")
    root, out = a.data_root.resolve(), a.out.resolve()
    spec = yaml.safe_load(a.run_list.read_text())
    runs = [r for r in spec["runs"] if not a.only
            or str(r["seed"]) in a.only or r.get("dataset") in a.only
            or spec["folder"].format(**r) in a.only]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(REPO)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
    verdicts: Dict[str, Any] = {}
    rc = 0
    for run in runs:
        folder = spec["folder"].format(**run)
        cmd = fit_argv(spec, run, root, out)
        log.info("%s: %s", folder, " ".join(cmd[1:]))
        if a.dry_run:
            continue
        if subprocess.run(cmd, cwd=REPO, env=env).returncode != 0:
            log.error("%s: fit failed", folder)
            return 1
        if a.no_compare:
            continue
        res = cmp.compare(out / folder, root / "models" / folder)
        verdicts[folder] = {"verdict": res["verdict"], **res["summary"]}
        log.info("%s: %s (%d/%d identical, max|d| %.3e)", folder, res["verdict"],
                 res["summary"]["n_identical"], res["summary"]["n_compared"],
                 res["summary"]["max_abs"])
        rc = max(rc, 3 if res["rc"] else 0)
    if verdicts:
        p = out / f"verdicts_{spec['name']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(verdicts, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
