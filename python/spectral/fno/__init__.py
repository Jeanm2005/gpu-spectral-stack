"""The Fourier Neural Operator on the Phase 2 flow data (Phase 3).

fno.model      FNO2d, SpectralConv2d and its backends, the relative L2 error
fno.config     training configuration: YAML files in configs/fno/ plus overrides
fno.tracking   experiment tracking: MLflow, or plain JSON files
fno.train      the training loop: python -m spectral.fno.train --config ...
"""
