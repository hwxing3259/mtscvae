# Benchmark

This folder holds the scorer of the paper's seven benchmark metrics and the two notebooks that
reproduce Fig 2 and Fig 3 from pre-computed numbers. The embeddings, the scores and the figure data
are in the [Figshare](https://doi.org/10.6084/m9.figshare.34971253) 

**Example: scoring HC embeddings.** The folder names are an example; any location works.
Put this repository, our figshare item and scMultiNODE's processed data
(doi:10.6084/m9.figshare.27420657.v2, one file, `data.zip`) side by side:

```
~/work/
├── mtscvae/                    this repository; run score.py from here
├── figshare/                   our figshare item: benchmark.zip and data.zip, unzipped here
└── scmultinode/                scMultiNODE's data.zip, unzipped here 
    └── data/
        ├── human_prefrontal_cortex_multiomic/reduce_processed/    HC
        │   ├── all-RNA-cell_meta.csv     all-RNA-data-hvg.csv
        │   └── all-ATAC-cell_meta.csv    all-ATAC-data-hvg.csv
        ├── human_organoid_Fleck2022/reduce_processed/             HO
        ├── drosophila_embryonic/reduce_processed/                 DR
        └── Yuan2022_MouseNeocortex/reduce_processed/              MN
```

```bash
export MTVAE_DATA=~/work/figshare
export SNR_DATA_ROOT=~/work/scmultinode           # the folder that holds data/; HC, HO, DR, MN only
python benchmark/score.py $MTVAE_DATA/benchmark/embeddings/HC/Ours_s0.npz       # benchmarking the seed 0 fit
```

