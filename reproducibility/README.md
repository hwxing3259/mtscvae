# Reproducing the paper's fits

From config files in [`mtvae/configs/`](../mtvae/configs) run:

```bash
# e.g. PFC data 
export MTVAE_DATA=~/work/figshare. # or where the datasets live
python -m mtvae.fit --config mtvae/configs/pfc.yaml --seed 44 --data-root $MTVAE_DATA --arm pfc_full/s44 --out-root ~/work/runs
# or HC data
python -m mtvae.fit --config mtvae/configs/fourds.yaml --seed 0 --dataset HC --data-root $MTVAE_DATA --arm fourds/HC/s0 --out-root ~/work/runs
```

The command trains Stage A and Stage B in one run and writes one fit folder.
That folder has the same layout as the pretrained models in the figshare item
(`models/<folder>`):
- `model.pt`, `latents/`;
- `stage_a.pt`, `stage_a_latents.npz`;
- `run_record.json`.
