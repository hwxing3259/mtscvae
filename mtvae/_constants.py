"""Package-wide constants.
The feature dimensions apply only to the PFC ``phase2_fusion`` checkpoint format."""
from __future__ import annotations

#: Default compute device when neither the CLI nor the config names one.
DEVICE: str = "cpu"

#: Shared latent dimension of both examples.
LATENT_DIM: int = 50

#: Feature counts of the PFC example (5,000 HVGs; 5,000 variable peaks).
INPUT_DIM_RNA: int = 5000
INPUT_DIM_ATAC: int = 5000

#: The six developmental stages of the PFC example, in developmental order.
STAGES_ORDERED: tuple[str, ...] = (
    "Gestation", "Neonatal", "Infancy", "Childhood", "Adolescence", "Adulthood")
