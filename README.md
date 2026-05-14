# Cascading Robustness Verification (CRV)

This repository contains the implementation for **Cascading Robustness Verification (CRV)**, introduced in our paper **Cascading Robustness Verification: Toward Efficient Model-Agnostic Certification**, accepted at **IEEE SaTML 2026**.

## Overview

Cascading Robustness Verification (CRV) verifies each input using a sequence of verifiers ordered from faster (lower compute) to slower (higher compute). For every input, CRV first applies the least expensive verifier. If the input is certified as robust, the pipeline stops early and moves to the next input. If not, the input is passed to the next verifier in the cascade.

This early-exit strategy avoids running expensive verifiers on inputs that can already be certified by cheaper methods. After all inputs are processed, the certified robust accuracy is computed as the fraction of inputs verified as robust by at least one verifier in the cascade.

## Requirements

The code has the following dependencies:

- Python 3.7
- MATLAB R2023b
- MOSEK 9.1.9 with an academic license
- CVX 2.2, Build 1148
- YALMIP

## Setup

Create two virtual environments, for example `lp` and `sdp`, for running the corresponding methods.

In the `lp` environment, install:

```bash
pip install -r convex_adversarial/requirements.txt
```

In the `sdp` environment, install:

```bash
pip install -r sdp/requirements.txt
```

Place the CVX and YALMIP folders inside the `sdp/` directory.

All MNIST inputs for which CRV should be run must be placed inside:

```text
custom_inputs/
```

Each input filename must strictly follow the format:

```text
<filename>_<true_label>.npy
```

Update the paths in the following files based on the absolute paths in your system:

```text
sdp/code/matlab_sdp.m
run_sdp.sh
certify.py
```

These paths should point to the correct locations of CVX, YALMIP, SDP inputs, virtual environments, and MATLAB.

## Running the Code

Finally, from the `lp` virtual environment, run:

```bash
python main.py --model_path convex_adversarial/models/mnist_model.pth
```

**Note:** The model must also be present in TensorFlow checkpoint format inside `sdp/models/` for the SDP verifier. This includes files such as `.meta`, `.index`, and the corresponding checkpoint data files. This is in accordance with the original implementation.

## Notes

- The SDP component requires MATLAB, CVX, YALMIP, and MOSEK to be configured correctly.
- The input label is parsed from the filename, so the required filename format must be followed.
- The local paths in the MATLAB and shell scripts must be updated before running the code.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{maleki2026cascadingrobustnessverificationefficient,
    title={Cascading Robustness Verification: Toward Efficient Model-Agnostic Certification}, 
    author={Mohammadreza Maleki and Rushendra Sidibomma and Arman Adibi and Reza Samavi},
    year={2026},
    eprint={2602.04236},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2602.04236}, 
}
```
