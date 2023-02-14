# Current Version: v0.2.2

## v0.2.2 (Feb 14, 2023)
### Added

- the following activation layers:
  - Identity
  - Binary Step
  - GELU
  - Softplus
  - ELU
  - SELU
  - Leaky ReLU
  - PReLU
  - SiLU
  - Gaussian
- a new example (Tic Tac Toe AI)

### Fixed

- a bug where setting `test=False` will return an error

## v0.2.1 (Feb 10, 2023)
### Added

- testing for RNNs
- moved `SGRU` to be a child of the `GRU` class
- renamed the orignial `SGRU` to be `OSGRU`
- allowed users to create their own GRU using The `GRU` as a base class

## v0.2 (Feb 10, 2023)
### Added

- testing for FFNNs
- visualizing for FFNNs
- an new example ([XOR](https://github.com/pleituer/neuralNet/tree/main/examples/XOR))

### Fixed

the visualizer (previously `test` function)

## v0.1 (Feb 8, 2023)
### Added

- Dense Layer
- Activations (tanh, sigmoid, ReLu, softmax)
- GRUs (a simplistic one with one internal weight and one internal activation function, tanh, and its called `SGRU`)
- LSTM (still in progress of debugging)
- FFNNs
- RNNs
