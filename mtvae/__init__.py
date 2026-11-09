"""mtvae: a multimodal temporal VAE with a latent SDE.

Stage A fits a two-modality VAE with a shared latent space; Stage B fits a
latent SDE across developmental stages while co-adapting the encoder and both
decoders. See the README for a walk-through.

Public API
----------
Dataset, Modality, stage_times_from
    The data contract the model consumes.
load_example, load_model, model_dir, data_root
    Read the paper's examples and fitted models from the figshare layout.
PipelineConfig, train
    Configure and fit both stages.
load_config
    Read a YAML config file.
"""
from mtvae.config import load as load_config
from mtvae.data import Dataset, Modality, stage_times_from
from mtvae.io import Model, data_root, load_example, load_model, model_dir
from mtvae.pipeline import PipelineConfig, build_dataset, train

__version__ = "0.1.0"

__all__ = ["Dataset", "Modality", "stage_times_from", "Model", "data_root",
           "load_example", "load_model", "model_dir", "PipelineConfig",
           "build_dataset", "train", "load_config", "__version__"]
