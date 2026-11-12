# mtvae: multimodal temporal integration with a latent SDE

`mtvae` integrates two single-cell modalities measured on *different* cells
(here snRNA-seq and snATAC-seq) across an ordered series of developmental stages,
and learns how cells move through the shared latent space over time.

* **Stage A** fits one variational autoencoder with modality-specific encoders
  and decoders and a shared stochastic fusion layer. The two modalities are
  matched stage by stage with a sliced-Wasserstein term, and each latent space is
  tilted toward the neighbourhoods of a fixed teacher embedding.
* **Stage B** fits a latent neural SDE to the stage-wise cohorts and co-adapts
  the encoder and decoders with it, held near the Stage-A solution by two anchors
  and regularised by a path KL term.

Flowing early cells through the fitted SDE and decoding them gives predicted
trajectories in feature space, for example a gene's expression across
development.

Paper: *link to be added*.  Data and trained models: figshare, *DOI to be added*.

![Method overview](docs/method_overview.png)

## Contents

| folder | what it holds |
|---|---|
| [`mtvae/`](mtvae) | the method: Stage A, Stage B, data contract, CLI (`python -m mtvae.fit`) |
| [`reproducibility/`](reproducibility) | run lists that regenerate every seeded fit in the paper from the figshare inputs, and the checks |
| [`tutorials/`](tutorials) | `pfc_seed44.ipynb`: load the PFC model, encode, latent UMAP, predicted gene trajectories |
| [`benchmark/`](benchmark) | the seven benchmark metrics, a scoring CLI, and the two notebooks that draw the paper's benchmark and PFC figures from pre-computed numbers |
| [`tests/`](tests) | a CPU smoke test |

## System requirements

### Software dependency and OS

* Python 3.10 or newer; tested with Python 3.12 (Linux, the GPU fits) and
  Python 3.11 (macOS, the tutorial and figures).
* torch, numpy, scipy, pandas, scikit-learn, openTSNE, PyYAML. The tutorial also
  needs umap-learn and matplotlib.
* Training runs on CPU or one NVIDIA GPU. The paper's fits used one GPU; the
  tutorial runs on a laptop CPU.
* `environment.yml` pins the exact stack the paper's fits were run with (torch
  2.8.0 + CUDA 12.8, numpy 2.1.2, scipy 1.18.0, scikit-learn 1.9.0, openTSNE
  1.0.4). Bit-for-bit reproduction needs that stack (see
  [`reproducibility/`](reproducibility)).

### Installation guide

```bash
pip install git+https://github.com/hwxing3259/mtscvae.git
```

or, for the pinned environment of the paper:

```bash
git clone https://github.com/hwxing3259/mtscvae.git && cd mtscvae
conda env create -f environment.yml && conda activate mtvae
```

Installation takes a few minutes. Then download the figshare item and point
`MTVAE_DATA` at the unzipped folder:

```bash
export MTVAE_DATA=/path/to/mtvae_figshare
```

## Demonstration

### Load relevant datasets

Every example reads the figshare layout (`data/`, `models/`, `benchmark/`,
`figures/`; see the figshare README).

```python
import mtvae

pfc = mtvae.load_example("pfc")                  # Herring et al. 2022 PFC, 6 stages
hc = mtvae.load_example("fourds", dataset="HC")  # one of the four benchmark datasets
print(pfc.names, [m.x.shape for m in pfc.modalities], pfc.stages)
```

### Define and train the model

One configuration covers both stages and the data handling. `for_example`
returns the recipe of the paper's fits (path-KL weight `beta_kl = 1e-4`). Any
field can be overridden, or the whole configuration read from a YAML file with
`mtvae.load_config`.

```python
cfg = mtvae.PipelineConfig.for_example("pfc", seed=44, device="cuda")
fit = mtvae.train(pfc, cfg)        # Stage A, then Stage B, in one call
z = fit.encode()                   # {"rna": (n_rna, 50), "atac": (n_atac, 50)}
```

The same from the command line. Every released fit was made by this one
command, from a config file in [`mtvae/configs/`](mtvae/configs):

```bash
python -m mtvae.fit --config mtvae/configs/pfc.yaml --seed 44 \
    --data-root $MTVAE_DATA --arm my_pfc_fit --out-root runs/
```

It writes one fit folder: `model.pt` (the final model, which loads on its
own), `latents/`, `stage_a.pt`, `stage_a_latents.npz` and `run_record.json`.

### Load a trained model, get latents and predicted trajectories

```python
from collections import OrderedDict

model = mtvae.load_model(mtvae.model_dir("pfc", seed=44))
z = model.encode(pfc)                                     # latents of every nucleus
astro = z["rna"][(pfc["rna"].stage == 0)][:300]           # some Gestation nuclei
traj = model.rollout(OrderedDict(cohort=astro), n_rollouts=5,
                     decode="rna", reduce=lambda x: x.mean(0))
expr = traj["cohort"].per_stage                           # 6 stages x 5,000 genes
```

[`tutorials/pfc_seed44.ipynb`](tutorials/pfc_seed44.ipynb) does this end to end:
the latent UMAP of both modalities and the predicted maturation of eight
astrocyte markers, checked against the paper's numbers.

### Estimated running time

* **Tutorial:** about 2 minutes on a laptop CPU.
* **PFC full atlas** (155k RNA and 87k ATAC nuclei): about 26 minutes on one
  RTX PRO 4500, both stages, of which Stage A takes about 3.
* **One four-dataset fit:** 14 to 16 minutes on one RTX 4090, both stages.
* **PFC at 8,000 nuclei per modality:** about 12 minutes on one RTX 4090, both
  stages.

## Instructions for use

**You provide** a `Dataset` of exactly two modalities (see `mtvae/data.py`):

* `Modality.x`: a cells × features float32 matrix, already normalised and
  scaled;
* `Modality.stage`: an integer stage per cell, in an ordered stage vocabulary;
* optionally:
  * `teacher`: a cells × k embedding for the neighbourhood tilt (needed when
    the tilt weight is above 0);
  * `train`: a training mask;
  * `trunk`: `TrunkSpec("mlp")`, or `TrunkSpec("spatial", chr_ids=...)` for the
    chromosome-aware ATAC encoder;
* `Dataset.stage_times`: normalised times of the stages in [0, 1]
  (`mtvae.stage_times_from` derives them from per-cell ages).

**Teacher embeddings.** For the tilt, pass a low-dimensional embedding of each modality, such
as a 100-d PCA of the normalised RNA or an LSI of the ATAC peaks, rather than the feature
matrix itself; the pipeline builds the neighbour affinities from it.

**You get back:**

* a shared 50-dimensional latent per cell;
* the fitted latent SDE;
* the two decoders.

Together these give stage-to-stage trajectories of any cell cohort, in latent
space or decoded to features.

## Reproducing the numerical examples in the paper

* **Fits.** [`reproducibility/`](reproducibility) regenerates the paper's 17
  seeded fits (four-dataset ×12, PFC full atlas ×3, PFC 8k ×2) with the one
  command above, and checks each refit against the released one, tensor for
  tensor.
* **Tutorial.** [`tutorials/pfc_seed44.ipynb`](tutorials/pfc_seed44.ipynb)
  reproduces the PFC latent UMAP and the astrocyte trajectories (seed 44).
* **Benchmark figure.** [`benchmark/fig2_benchmark.ipynb`](benchmark/fig2_benchmark.ipynb)
  draws it, and the same values as a bar grid, from the pre-computed scores.
* **PFC figure.** [`benchmark/fig3_pfc.ipynb`](benchmark/fig3_pfc.ipynb) draws
  it and Table 1 from the pre-computed values.
* **Scoring.** [`benchmark/`](benchmark) explains how to re-score any
  embedding.

## Pre-trained models and datasets

The figshare item (*DOI to be added*) holds:

* the processed inputs of both examples;
* all 18 released fit folders (`model.pt`, latents, Stage-A checkpoint and run
  record): four-dataset seeds 0–2, PFC full atlas seeds 22/44/88, PFC 8k
  seeds 22/44/88;
* every benchmark embedding and score;
* the values behind the paper's figures.

The four-dataset inputs are re-hosted from scMultiNODE's processed release
(doi:10.6084/m9.figshare.27420657.v2). The PFC data are from Herring et al.
(2022), *Cell* 185:4428.

## License

MIT; see [LICENSE](LICENSE).
