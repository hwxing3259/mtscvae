# Benchmark

This folder holds the scorer of the paper's seven benchmark endpoints and the two notebooks that
draw Fig 2 and Fig 3 from pre-computed numbers. The embeddings, the scores and the figure data
are in the figshare item (`benchmark/` and `figures/`). Data paths below are relative to the
unzipped figshare folder, `$MTVAE_DATA`. The notebooks default to `../figshare` next to the
repository.

```
benchmark/
├── score.py               scores embeddings with the scorer of record
├── metrics/               the scorer of record, as copies with short module docstrings. Machine
│                          paths in docstrings, unused path defaults and provenance strings
│                          are given as home folders (~/...); the code is otherwise identical
├── figures/               the figures' CSV-only plot modules, as copies with short module
│                          docstrings; one legend label of the runtime figure, not drawn here,
│                          is reworded
├── fig2_benchmark.ipynb   Fig 2 (dot plot), and the same values as a bar grid, from
│                          figures/fig2_benchmark/
└── fig3_pfc.ipynb         Fig 3 and Table 1 from figures/fig3_pfc/
```

"Ours" is the release configuration everywhere: the SDE path-KL at β = 1e-4.
* Four datasets: seeds 0, 1 and 2.
* PFC at the matched 8k: seeds 22, 44 and 88.
* PFC on the full split: seeds 22, 44 and 88.

## The seven endpoints

| endpoint | family | what it measures |
|---|---|---|
| `time_corr` | temporal consistency | distance correlation between each modality's embedding and the time point, averaged over the two modalities |
| `nmi` | latent structure | NMI between Leiden clusters and annotated cell types, per modality, averaged; cells labelled `unknown` are left out |
| `batch_entropy` | latent structure | prior-adjusted entropy (natural log) of the modality mix among the 100 nearest neighbours of sampled cells; at most ln 2 ≈ 0.69; higher is better mixed |
| `lta_time` | cross-modal transfer | label-transfer accuracy: a 50-NN classifier fitted on the ATAC cells predicts the RNA cells' time points |
| `lta_type` | cross-modal transfer | the same transfer for cell type, over the types both modalities annotate |
| `rna_rare_sil`, `atac_rare_sil` | rare-type resolution | scMultiNODE's rare-population rule: the one-vs-rest silhouette of each of the three rarest types (one on HO), averaged |

`nmi` is Leiden on every dataset, PFC included, because one scorer computes all of them. The PFC
score records also carry the PFC example's own panel (`pfc_*` keys). Its `pfc_*_nmi_kmeans` is a
k-means NMI (k = 20), which is not drawn in Fig 2 and is not comparable with `nmi`. Values are not
comparable across datasets; the ranks are.

## Scoring an embedding

An embedding is `benchmark/embeddings/<folder>/<Method>_s<seed>.npz`, holding the arrays `rna`
and `atac` (cells × 50). `<folder>` is HC, HO, DR, MN, PFC_8k or PFC_full. On HO, the Pamona and
UnionCom embeddings are scMultiNODE's published latents (doi:10.6084/m9.figshare.27418872.v4),
which its authors fitted on all cells. Their rows are put into the scorer's row order, the values
are unchanged, and they are scored like every other embedding. A fit on a subsample would carry
the suffix `_n<cells>`, and the scorer would re-draw that subsample from the suffix. None ships.

**Example: scoring HC and PFC embeddings.** The folder names are an example; any location works.
Put this repository, our figshare item and scMultiNODE's processed data
(doi:10.6084/m9.figshare.27420657.v2, one file, `data.zip`) side by side:

```
~/work/
├── mtscvae/                    this repository; run score.py from here
├── figshare/                   our figshare item: benchmark.zip and data.zip, unzipped here
│   ├── benchmark/embeddings/HC/Ours_s0.npz, ..., PFC_8k/Ours_s22.npz, ...
│   └── data/pfc/bench_tiers.npz, rna_metadata.csv, atac_metadata.csv
└── scmultinode/                scMultiNODE's data.zip, unzipped here (SNR_DATA_ROOT)
    └── data/
        ├── human_prefrontal_cortex_multiomic/reduce_processed/    HC
        │   ├── all-RNA-cell_meta.csv     all-RNA-data-hvg.csv
        │   └── all-ATAC-cell_meta.csv    all-ATAC-data-hvg.csv
        ├── human_organoid_Fleck2022/reduce_processed/             HO
        ├── drosophila_embryonic/reduce_processed/                 DR
        └── Yuan2022_MouseNeocortex/reduce_processed/              MN
```

```bash
cd ~/work
unzip figshare/benchmark.zip -d figshare          # -> figshare/benchmark/
unzip figshare/data.zip 'data/pfc/bench_tiers.npz' 'data/pfc/*_metadata.csv' -d figshare
                                                  # PFC only; the scorer reads just these three
unzip scmultinode/data.zip -d scmultinode         # -> scmultinode/data/ (24.7 GB unzipped)

cd mtscvae
export MTVAE_DATA=~/work/figshare
export SNR_DATA_ROOT=~/work/scmultinode           # the folder that holds data/; HC, HO, DR, MN only
python benchmark/score.py $MTVAE_DATA/benchmark/embeddings/HC/Ours_s0.npz       # one fit
python benchmark/score.py $MTVAE_DATA/benchmark/embeddings/HC/*.npz             # every HC fit
python benchmark/score.py $MTVAE_DATA/benchmark/embeddings/PFC_8k/Ours_s22.npz  # MTVAE_DATA only
```

Each embedding is written to `scores/<folder>/<stem>.json` under the current folder (`--out-dir`
changes it): here `scores/HC/Ours_s0.json` and `scores/PFC_8k/Ours_s22.json`. The seven endpoints
are top-level keys, next to the scorer's diagnostics. They should equal that fit's rows of
`benchmark/scores.csv` (`time_corr` to a few 1e-4, see **Exactness** below). For `HC/Ours_s0`
(dataset `HC`, tier `all`, method `Ours`, seed 0) they are `time_corr` 0.790661, `nmi` 0.671504,
`batch_entropy` 0.678066, `lta_time` 0.591568, `lta_type` 0.802372, `rna_rare_sil` 0.455437 and
`atac_rare_sil` 0.379246. Outside those versions (**Environment** below), `nmi` can move: with
scanpy 1.10.3, igraph 0.11.9 and leidenalg 0.10.2 this example gave 0.673772 (+0.0023), the graph
clustering behind `nmi` being version-dependent, and the other six as above.

`score.py` runs the scorer of record in a process of its own, with the same command line as the
published scores. Its stochastic steps are seeded with the fit's seed.
* **HC, HO, DR, MN** run `mtvae_4ds_coadapt.data.snr_score`, then add the two rare-type endpoints.
  They need scMultiNODE's processed data: doi:10.6084/m9.figshare.27420657.v2, `data.zip`,
  unpacked so that `$SNR_DATA_ROOT/data/<dataset folder>/reduce_processed/` exists.
* **PFC** runs `pfc_scaling_bench.score` on rows `[:8000]` of each modality, the evaluation draw:
  the 8,000 nuclei per modality that the matched tier is fitted on. Every PFC fit, at 8k or on
  the full split, is scored on these same nuclei.
  It reads only three files from `$MTVAE_DATA/data/pfc/`:
  * `bench_tiers.npz` (`perm_rna`, `perm_atac`);
  * `rna_metadata.csv` (`barcode`, `major_clust`, `stage_id`, `donor_id`);
  * `atac_metadata.csv` (`barcode`, `predicted_group`, `stage_id`, `donor_id`).

  It does not read the expression matrices.

**Environment.** Use the versions the scores were made with: the main README lists them, and
`environment.yml` installs them.
* The published scores were made on Linux. On macOS the scorer can crash in an OpenMP runtime
  clash.
* Scoring an HO embedding (10,000 cells per modality) takes about 33 GB of memory.

**Exactness.** With those versions, six endpoints re-score exactly. `time_corr`, a distance
correlation over all the cells scored, re-scores only to a few 1e-4 across machines (2.1e-4 seen on
a PFC embedding). The four-dataset rare-type values of record are kept at 6 dp.

## The scores in figshare

* **`benchmark/scores.csv`** has the columns `dataset, tier, method, seed, metric, value`: one
  row per fit and endpoint, unrounded. Read it with `pd.read_csv(..., float_precision="round_trip")`.
  `tier` is one of:
  * `all`: the dataset as released;
  * `n8000`: PFC matched;
  * `full`: PFC full training split.
* **`benchmark/scores/<folder>/`** holds the per-fit score records of record, which carry the nulls
  and diagnostics too.
* **`benchmark/scores/rare_type_fourds.csv`** holds the four-dataset rare-type values.
* **`benchmark/manifest.csv`** lists every embedding and what its rows are.

The means over seeds of `scores.csv` give the values of Fig 2 (`grid_values.csv`) and the bars of
Fig 3a (`8k_vs_full_values.csv`).

## Where the competitor embeddings come from

No competitor runner ships with this repository. The embeddings come pre-computed in figshare,
each with a `<stem>.json` beside it: run time, device and hyperparameters.
* **Runner of record.** All six methods (scMultiNODE, SCOTv1, SCOTv2, Pamona, UnionCom and
  uniPort) were run by `snr_run_method.py`, at its `BASELINE_HPARAMS`. Those are the methods'
  wrapper defaults, under scMultiNODE's experimental configuration, with a 50-d output.
  There is one exception: HO Pamona and UnionCom are scMultiNODE's own published latents
  (see above). Their `<stem>.json` gives the source file instead of a run.
* **scMultiNODE** used its per-dataset settings. On PFC it used its human-cortex (HC) settings.
* **Fits per method.**
  * Four datasets: scMultiNODE 5 seeds (4 on DR), uniPort 5, the others 1.
  * PFC: one fit each.

## Drawing the figures

Run the notebooks from this folder. `MTVAE_DATA` points at the figshare folder, and
`MTVAE_FIG_OUT` (optional) is where the PNG and SVG are written; the default is a temporary
folder. Each notebook reads only its `figures/` subfolder and draws with the modules in
`figures/`, without changing them. The figures were rendered with matplotlib 3.11.1.

## Read with care

* **The PFC benchmark is in-sample.** Every method is scored on training nuclei of the
  whole-donor split.
* **Fig 2's PFC values have 3 seeds of ours** and one fit per competitor.
* **DR `rna_rare_sil` is not drawn.** Its three rarest RNA types hold 1, 1 and 2 cells. It is
  still in `scores.csv`.
* **DR `lta_type` is 1.000 for every method.**
* **HO cell types are a coarsening of the time point**, so the type-based endpoints are circular
  with time there. HO is not a co-assay: its cells are computational metacells.
* **PFC `lta_type` is partly circular.** The ATAC cell types are label transfers from the RNA
  reference.
