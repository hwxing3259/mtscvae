# Reproducing the paper's fits

Every released fit was made by the same command, from one of three config files
in [`mtvae/configs/`](../mtvae/configs):

```bash
python -m mtvae.fit --config mtvae/configs/<config>.yaml --seed <s> [--dataset <DS>] \
    --data-root $MTVAE_DATA --arm <folder> --out-root <out>
```

The command trains Stage A and Stage B in one run and writes one fit folder.
That folder has the same layout as the released fits in the figshare item
(`models/<folder>`):
- `model.pt`, `latents/`;
- `stage_a.pt`, `stage_a_latents.npz`;
- `run_record.json`.

| run list | config | fits | released folders |
|---|---|---|---|
| `fourds.yaml` | `fourds.yaml` | HC, HO, DR, MN × seeds 0, 1, 2 (Fig 2) | `models/fourds/<DS>/s<seed>` |
| `pfc_full.yaml` | `pfc.yaml` | PFC full atlas, seeds 22, 44, 88 (Fig 3; tutorial: 44) | `models/pfc_full/s<seed>` |
| `pfc_8k.yaml` | `pfc_8k.yaml` | PFC, 8,000 nuclei per modality, seeds 22, 44, 88 (Fig 2, PFC column; Fig 3a) | `models/pfc_8k/s<seed>` |

Each config holds every setting of its runs:
- the data handling: input clipping, training split, row subset;
- the encoder trunks;
- Stage A;
- the Stage-B objective, including the path-KL weight `beta_kl = 1e-4`;
- the SDE geometry.

The configs differ only where the paper's fits differ:
- **Stage-A recipe:** four-dataset versus PFC.
- **The ATAC encoder:** an MLP for the four datasets, the chromosome-aware Conv1d for PFC.
- **The data handling:**
  - four-dataset: ATAC clipped at ±5;
  - PFC full atlas: RNA unclipped and ATAC clipped at ±10, reconstruction targets at ±10, whole-donor training split;
  - PFC 8k: both clipped at ±10, the first 8,000 rows of the benchmark order.

## Running and checking

```bash
export MTVAE_DATA=/path/to/mtvae_figshare
python reproducibility/run_all.py reproducibility/pfc_full.yaml --out runs/ --only 44
python reproducibility/run_all.py reproducibility/fourds.yaml --out runs/          # all 12
```

`run_all.py` runs the listed fits one at a time. It then compares each refit with
its released folder, tensor for tensor and with no tolerance (`compare.py`): the
SDE, encoder and decoders in `model.pt`, both latents and `stage_norm`, `stage_a.pt`
and the Stage-A latents.
- `--dry-run` prints the commands.
- `compare.py` also works on its own: `python reproducibility/compare.py runs/pfc_full/s44 $MTVAE_DATA/models/pfc_full/s44`.

**Bit-exact reproduction needs the GPU model and the software stack of the
released fits.** The stack is pinned in `../environment.yml`: torch 2.8.0 +
CUDA 12.8, numpy 2.1.2, scikit-learn 1.9.0, scipy 1.18.0, openTSNE 1.0.4. cuDNN
determinism is on by default. The GPUs:

| fits | GPU |
|---|---|
| four-dataset, PFC 8k | NVIDIA RTX 4090 |
| PFC full atlas | NVIDIA RTX PRO 4500 (Blackwell) |

Every fit, on either GPU, is one run of the command above.

The floating-point result depends on the GPU architecture: different
architectures run different cuDNN and cuBLAS kernels, notably for the PFC
model's chromosome-aware Conv1d encoder. On another GPU, or on CPU, expect
differences at float precision from the first iterations, and from there a
different fit of the same recipe.

As a measure of the size: for the PFC full-atlas fits, running Stage B on an
RTX 4090 instead of the RTX PRO 4500 changes the seven benchmark endpoints by
at most 0.002 (`nmi` by up to 0.006) and keeps 96 to 99% of each nucleus's 15
nearest neighbours in the latent. The astrocyte rising margins of Fig 3 move by
up to 0.18 for one seed, about the spread between seeds.

## Timing

- **PFC full-atlas fit,** on one RTX PRO 4500: about 26 minutes, of which Stage A is about 3.
- **Four-dataset fit,** on one RTX 4090: 14 to 16 minutes.
- **PFC 8k fit,** on one RTX 4090: about 12 minutes.
