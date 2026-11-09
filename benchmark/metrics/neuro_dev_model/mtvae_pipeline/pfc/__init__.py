"""PFC (Herring 2022) lab — isolated add-ons for the human prefrontal-cortex dataset.

This package is **additive and self-contained**.  It imports the shipped
pipeline and ``vae`` read-only and edits nothing in either.  Rollback is
``rm -rf pfc/``.

Contents
--------
gumbel_gate
    Binary-concrete (Gumbel-sigmoid) sparsity gate for Phase 3, as a *subclass*
    of the shipped ``SparsePerturbationHead``.  The shipped class is untouched.
"""
