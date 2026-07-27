# ASCA-AD V4-IO2

V4-IO2 is an independent, inference-only experiment. It preserves the 146
checkpoint parameters and the verified V4-IO total-score mathematics while
processing lag candidates in chunks. Channel-heavy gather tensors are reduced
immediately and are never concatenated into a full `[B,L,K,C]` tensor.

This package does not contain training code and does not create checkpoints.
Formal adoption is gated by pointwise score, threshold, RAW, PA, speed and GPU
memory equivalence tests in `run_asca_v4_io2_optimization.py`.
