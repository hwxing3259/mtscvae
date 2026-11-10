"""mtvae: a multimodal temporal VAE with a latent SDE.
Stage A fits a two-modality VAE with a shared latent; Stage B fits a latent SDE across stages."""
from mtvae.config import load as load_config
from mtvae.data import Dataset, Modality, stage_times_from
from mtvae.io import Model, data_root, load_example, load_model, model_dir
from mtvae.pipeline import PipelineConfig, build_dataset, train

__version__ = "0.1.0"

__all__ = ["Dataset", "Modality", "stage_times_from", "Model", "data_root",
           "load_example", "load_model", "model_dir", "PipelineConfig",
           "build_dataset", "train", "load_config", "__version__"]
