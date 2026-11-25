# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-25

First public release of a sim-to-real RL stack for a DIY 12-DoF quadruped, with a first walking policy.

### Added
- Parameterised MJCF model, vectorised MuJoCo env, domain randomisation, DC motor model
- PPO from scratch in PyTorch; TorchScript and ONNX export
- 50 Hz robot runtime with a safety layer and an MIT-mode CAN driver; sim-in-the-loop mode
- `walk_v0` policy, trained 77 min on a desktop CPU: about 0.09 m/s tracking error at a 0.5 m/s command with randomisation, no falls (simulation only)

[0.1.0]: https://github.com/coopersimpson101-debug/dog2real/releases/tag/v0.1.0
