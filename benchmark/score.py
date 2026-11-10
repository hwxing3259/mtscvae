"""Score benchmark embeddings with the scorer of record.
Example: ``python benchmark/score.py $MTVAE_DATA/benchmark/embeddings/HC/Ours_s0.npz``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("score")

HERE: Path = Path(__file__).resolve().parent
METRICS: Path = HERE / "metrics"
FOURDS: tuple[str, ...] = ("HC", "HO", "DR", "MN")
#: PFC embedding folders: (tier label, cells per modality in the fit).
PFC: Dict[str, tuple[str, Optional[int]]] = {"PFC_8k": ("n8000", 8000), "PFC_full": ("full", None)}
ENDPOINTS: tuple[str, ...] = ("time_corr", "nmi", "batch_entropy", "lta_time", "lta_type",
                              "rna_rare_sil", "atac_rare_sil")
RARE: tuple[str, ...] = ("rna_rare_sil", "atac_rare_sil")
#: OpenMP/MKL threads per scorer process, as the published scores were made.
THREADS: Dict[str, int] = {"HC": 4, "DR": 4, "MN": 4, "HO": 8, "PFC_8k": 5, "PFC_full": 5}
#: PFC evaluation draw E: the first N_EVAL rows of each modality's tier permutation, which
#: are the cells the matched tier is fitted on. Passed to the scorer as ``PFCSB_N_EVAL``.
N_EVAL: int = 8000
PFC_META_COLUMNS: Dict[str, tuple[str, ...]] = {
    "rna": ("barcode", "major_clust", "stage_id", "donor_id"),
    "atac": ("barcode", "predicted_group", "stage_id", "donor_id")}
STEM = re.compile(r"^(?P<method>[A-Za-z0-9]+)_s(?P<seed>\d+)(?:_n(?P<subsample>\d+))?$")
FOURDS_MODULE = "neuro_dev_model.mtvae_4ds_coadapt.data.snr_score"
PFC_MODULE = "neuro_dev_model.pfc_scaling_bench.score"


@dataclass(frozen=True)
class Embedding:
    """One embedding file of the figshare layout."""

    path: Path
    folder: str
    method: str
    seed: int
    subsample: Optional[int]

    @property
    def stem(self) -> str:
        """``<Method>_s<seed>[_n<subsample>]``."""
        return self.path.stem


def parse(path: Path) -> Embedding:
    """Read dataset, method, seed and subsample from an embedding's path.

    Parameters
    ----------
    path : pathlib.Path
        ``.../benchmark/embeddings/<folder>/<Method>_s<seed>[_n<subsample>].npz``.

    Returns
    -------
    Embedding
        The parsed embedding.

    Raises
    ------
    ValueError
        If the folder or the file name does not follow the layout.
    """
    folder = path.parent.name
    m = STEM.match(path.stem)
    if path.suffix != ".npz" or m is None or folder not in FOURDS + tuple(PFC):
        raise ValueError(f"{path}: expected embeddings/<HC|HO|DR|MN|PFC_8k|PFC_full>/"
                         "<Method>_s<seed>[_n<subsample>].npz")
    sub = m.group("subsample")
    if sub and folder in PFC:
        raise ValueError(f"{path}: PFC embeddings carry no subsample")
    return Embedding(path, folder, m.group("method"), int(m.group("seed")),
                     int(sub) if sub else None)


def figshare_root(emb: Embedding, data_root: Optional[Path]) -> Path:
    """The figshare folder: ``--data-root``, ``$MTVAE_DATA``, or the embedding's own.

    Parameters
    ----------
    emb : Embedding
        An embedding inside ``<figshare>/benchmark/embeddings/``.
    data_root : pathlib.Path, optional
        Explicit root.

    Returns
    -------
    pathlib.Path
        The figshare root.
    """
    if data_root is not None:
        return data_root
    if os.environ.get("MTVAE_DATA"):
        return Path(os.environ["MTVAE_DATA"])
    return emb.path.resolve().parents[3]


def check_embedding(emb: Embedding) -> Dict[str, tuple[int, int]]:
    """Check an embedding holds the two arrays the scorer reads.

    Parameters
    ----------
    emb : Embedding
        The embedding.

    Returns
    -------
    dict
        Shape per modality.

    Raises
    ------
    ValueError
        If an array is missing, not 2-D, or a PFC embedding is shorter than E.
    """
    with np.load(emb.path) as z:
        shapes = {m: z[m].shape for m in ("rna", "atac") if m in z.files}
    if set(shapes) != {"rna", "atac"} or any(len(s) != 2 for s in shapes.values()):
        raise ValueError(f"{emb.path}: needs 2-D arrays 'rna' and 'atac', has {shapes}")
    if emb.folder in PFC and min(s[0] for s in shapes.values()) < N_EVAL:
        raise ValueError(f"{emb.path}: a PFC embedding needs at least {N_EVAL} rows per modality")
    return shapes


def scorer_env(threads: int, extra: Dict[str, str]) -> Dict[str, str]:
    """Environment of a scorer process, as the published scores were made.

    Parameters
    ----------
    threads : int
        OpenMP / MKL threads.
    extra : dict
        Further variables (data roots).

    Returns
    -------
    dict
        The environment.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(METRICS), env.get("PYTHONPATH")]))
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(threads)
    env["NUMBA_THREADING_LAYER"] = "workqueue"
    env.update(extra)
    return env


def run(cmd: List[str], env: Dict[str, str], cwd: Path, log_path: Path) -> None:
    """Run one scorer process, its output going to a log file.

    Parameters
    ----------
    cmd : list of str
        Command line.
    env : dict
        Environment.
    cwd : pathlib.Path
        Working directory.
    log_path : pathlib.Path
        Where stdout and stderr go.

    Raises
    ------
    RuntimeError
        If the process fails; the end of its log is included.
    """
    with open(log_path, "w") as fh:
        rc = subprocess.run(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "".join(log_path.read_text().splitlines(keepends=True)[-20:])
        raise RuntimeError(f"scorer exited with {rc}:\n{tail}")


def rare_type(emb: Embedding, snr_root: Path) -> Dict[str, Any]:
    """scMultiNODE's rare-population silhouette per modality (the adopted rule).

    The same calls as ``run_benchmark.rare_type`` of the four-dataset deposit.

    Parameters
    ----------
    emb : Embedding
        A four-dataset embedding.
    snr_root : pathlib.Path
        scMultiNODE data root.

    Returns
    -------
    dict
        ``{m}_rare_sil`` and the rule's components, per modality.
    """
    os.environ["SNR_DATA_ROOT"] = str(snr_root)  # read by snr_config at import
    if str(METRICS) not in sys.path:
        sys.path.insert(0, str(METRICS))
    from neuro_dev_model.mtvae.data import snr_data
    from neuro_dev_model.mtvae.data import snr_rare_population as rare

    data = snr_data.load_dataset(emb.folder, mode="harmonised")
    if emb.subsample:
        data = snr_data.subsample(data, emb.subsample, seed=emb.seed)  # the cells of the fit
    out: Dict[str, Any] = {}
    with np.load(emb.path) as z:
        for m in ("rna", "atac"):
            col = snr_data.REFERENCE_TYPE_COL[emb.folder][m]
            lab = rare.remap_unknown(
                np.asarray([str(x).lower() for x in data.modality(m).meta[col]]), emb.folder)
            lat = np.asarray(z[m], dtype=np.float64)
            if len(lat) != len(lab):
                raise RuntimeError(f"{emb.path} {m}: {len(lat)} rows vs {len(lab)} labels")
            rp = rare.rare_population(lat, lab, emb.folder)
            out[f"{m}_rare_sil"] = float(rp["silhouette"])
            for k in ("rare_types", "rare_counts", "per_type_silhouette", "rare_pct"):
                out[f"{m}_rare_{k}"] = rp[k]
    return out


def score_fourds(emb: Embedding, snr_root: Path, threads: int, work: Path,
                 dry_run: bool) -> Optional[Dict[str, Any]]:
    """Score a four-dataset embedding.

    Parameters
    ----------
    emb : Embedding
        The embedding.
    snr_root : pathlib.Path
        scMultiNODE data root, holding ``data/<dataset folder>/reduce_processed/``.
    threads : int
        Scorer threads.
    work : pathlib.Path
        Scratch folder.
    dry_run : bool
        Check the inputs and log the command only.

    Returns
    -------
    dict or None
        The score record, or ``None`` on a dry run.
    """
    if not (snr_root / "data").is_dir():
        raise FileNotFoundError(f"{snr_root}/data not found: set --snr-data-root or "
                                "SNR_DATA_ROOT to the unpacked scMultiNODE data")
    cmd = [sys.executable, "-m", FOURDS_MODULE, "--dataset", emb.folder,
           "--method", emb.method, "--seed", str(emb.seed),
           "--runs-dir", str(emb.path.parent.resolve()), "--out-dir", str(work), "--force"]
    if emb.subsample:
        cmd += ["--subsample", str(emb.subsample)]
    log.info("%s: %s", emb.stem, " ".join(cmd[1:]))
    if dry_run:
        return None
    run(cmd, scorer_env(threads, {"SNR_DATA_ROOT": str(snr_root.resolve())}), work,
        work / "scorer.log")
    out = work / f"{emb.stem}.json"
    if not out.is_file():  # the scorer exits 0 without writing when it finds no npz
        raise RuntimeError(f"the scorer wrote no record for {emb.path}")
    rec = json.loads(out.read_text())
    rec.update(rare_type(emb, snr_root.resolve()))
    return rec


def score_pfc(emb: Embedding, data_root: Path, threads: int, work: Path,
              dry_run: bool) -> Optional[Dict[str, Any]]:
    """Score a PFC embedding through the PFC scorer of record.

    Parameters
    ----------
    emb : Embedding
        The embedding.
    data_root : pathlib.Path
        The figshare root, holding ``data/pfc/``.
    threads : int
        Scorer threads.
    work : pathlib.Path
        Scratch folder, used as the scorer's output root.
    dry_run : bool
        Check the inputs and log the command only.

    Returns
    -------
    dict or None
        The score record, or ``None`` on a dry run.

    Raises
    ------
    FileNotFoundError, ValueError
        If ``data/pfc/`` lacks a file, a column or a tier key the scorer reads.
    """
    pfc = data_root / "data" / "pfc"
    tiers = pfc / "bench_tiers.npz"
    for f in (tiers, *(pfc / f"{m}_metadata.csv" for m in PFC_META_COLUMNS)):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found: set --data-root or MTVAE_DATA")
    with np.load(tiers) as z:
        if not {"perm_rna", "perm_atac"} <= set(z.files):
            raise ValueError(f"{tiers}: needs perm_rna and perm_atac, has {z.files}")
    for m, cols in PFC_META_COLUMNS.items():
        have = pd.read_csv(pfc / f"{m}_metadata.csv", nrows=0).columns
        if set(cols) - set(have):
            raise ValueError(f"{m}_metadata.csv lacks {sorted(set(cols) - set(have))}")
    tier, n_tier = PFC[emb.folder]
    tag = f"{emb.method}_{tier}_s{emb.seed}"
    cmd = [sys.executable, "-m", PFC_MODULE, "--tag", tag, "--force"]
    log.info("%s: %s", emb.stem, " ".join(cmd[1:]))
    if dry_run:
        return None
    for sub in ("emb", "records", "scores"):
        (work / sub).mkdir()
    (work / "tiers.npz").symlink_to(tiers.resolve())
    (work / "emb" / f"{tag}.npz").symlink_to(emb.path.resolve())
    (work / "records" / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "method": emb.method, "tier": tier, "n_tier": n_tier,
         "seed": emb.seed, "status": "ok"}))
    env = scorer_env(threads, {"PFCSB_DATA_DIR": str(pfc.resolve()), "PFCSB_OUT": str(work),
                               "PFCSB_TRAIN_MASK": str((pfc / "wholedonor_train_mask.npz").resolve()),
                               "PFCSB_N_EVAL": str(N_EVAL)})
    run(cmd, env, work, work / "scorer.log")
    rec = json.loads((work / "scores" / f"{tag}.json").read_text())
    if rec.get("n_eval") != N_EVAL:
        raise RuntimeError(f"{emb.path}: scored on {rec.get('n_eval')} rows, not {N_EVAL}")
    return rec


def compare(emb: Embedding, rec: Dict[str, Any], table: pd.DataFrame,
            tol_time_corr: float, tol: float = 0.0) -> bool:
    """Compare the seven endpoints with the matching rows of ``scores.csv``.

    Parameters
    ----------
    emb : Embedding
        The embedding scored.
    rec : dict
        Its new score record.
    table : pandas.DataFrame
        ``benchmark/scores.csv``.
    tol_time_corr : float
        Tolerance for ``time_corr``, which re-scores only to a few 1e-4 across machines
        (2.1e-4 seen on a PFC embedding of 8,000 + 8,000 nuclei).
        Four-dataset rare-type values are carried at 6 dp, so they get 5e-7. The other
        endpoints must match exactly.
    tol : float, optional
        A floor on every endpoint's tolerance, for refitted rather than re-scored
        embeddings (the PFC 8k refits land within 0.0015 of the record).

    Returns
    -------
    bool
        True if every endpoint is within its tolerance.
    """
    dataset = "PFC" if emb.folder in PFC else emb.folder
    tier = PFC[emb.folder][0] if emb.folder in PFC else (f"n{emb.subsample}" if emb.subsample else "all")
    ref = table[(table.dataset == dataset) & (table.tier == tier) & (table.method == emb.method)
                & (table.seed == emb.seed)].set_index("metric").value
    if ref.empty:
        log.warning("%s: no rows in scores.csv", emb.stem)
        return False
    ok = True
    for ep in ENDPOINTS:
        tol = max(tol_time_corr if ep == "time_corr"
                  else (5e-7 + 1e-12 if dataset in FOURDS and ep in RARE else 0.0), tol)
        d = abs(float(rec[ep]) - float(ref[ep]))
        ok &= d <= tol
        log.info("%s %-14s new %.10f  record %.10f  |diff| %.1e  %s", emb.stem, ep,
                 float(rec[ep]), float(ref[ep]), d, "ok" if d <= tol else "DIFFERS")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score each embedding given, one at a time.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.

    Returns
    -------
    int
        0 if every embedding was scored (and, with ``--compare``, matched).
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("embeddings", nargs="+", type=Path, help="embedding .npz files")
    p.add_argument("--data-root", type=Path, default=None,
                   help="figshare folder (default: $MTVAE_DATA, else the embedding's own)")
    p.add_argument("--snr-data-root", type=Path, default=os.environ.get("SNR_DATA_ROOT"),
                   help="scMultiNODE data root for HC/HO/DR/MN (default: $SNR_DATA_ROOT)")
    p.add_argument("--out-dir", type=Path, default=Path("scores"),
                   help="writes <out-dir>/<folder>/<stem>.json (default: ./scores)")
    p.add_argument("--threads", type=int, default=None,
                   help="scorer threads (default: as published, 4 on HC/DR/MN, 8 on HO, 5 on PFC)")
    p.add_argument("--compare", type=Path, default=None, help="benchmark/scores.csv to check against")
    p.add_argument("--tol-time-corr", type=float, default=5e-4)
    p.add_argument("--tol", type=float, default=0.0,
                   help="tolerance floor for every endpoint, e.g. 0.0015 for a refitted embedding")
    p.add_argument("--dry-run", action="store_true", help="check the inputs, run nothing")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    table = pd.read_csv(args.compare, float_precision="round_trip") if args.compare else None
    failed = 0
    for path in args.embeddings:
        emb = parse(path)
        shapes = check_embedding(emb)
        log.info("%s/%s: rna %s, atac %s", emb.folder, path.name, shapes["rna"], shapes["atac"])
        threads = args.threads or THREADS[emb.folder]
        with tempfile.TemporaryDirectory(prefix="score_") as tmp:
            work = Path(tmp)
            if emb.folder in PFC:
                rec = score_pfc(emb, figshare_root(emb, args.data_root), threads, work, args.dry_run)
            else:
                if args.snr_data_root is None:
                    p.error("HC/HO/DR/MN need --snr-data-root or SNR_DATA_ROOT")
                rec = score_fourds(emb, Path(args.snr_data_root), threads, work, args.dry_run)
        if rec is None:
            continue
        rec["embedding"] = f"{emb.folder}/{path.name}"
        out = args.out_dir / emb.folder / f"{emb.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, default=str))
        log.info("%s: %s -> %s", emb.stem,
                 {k: round(float(rec[k]), 4) for k in ENDPOINTS}, out)
        if table is not None and not compare(emb, rec, table, args.tol_time_corr, args.tol):
            failed += 1
    if failed:
        log.error("%d embedding(s) differ from %s", failed, args.compare)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
