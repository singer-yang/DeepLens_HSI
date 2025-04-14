# DeepLens HSI: Hyperspectral Imaging with DeepLens

This repository provides code for end-to-end hyperspectral imaging (HSI) simulation and reconstruction using the [DeepLens](https://github.com/singer-yang/DeepLens) framework. It demonstrates how to model diffractive optical elements (DOEs) for HSI and train deep learning models to recover spectral information from simulated RGB sensor captures.

![HSI Reconstruction Demo](./video1_raw.gif)

## Features

DeepLens HSI offers a fully differentiable pipeline enabling:

- **Optics Simulation:** Model diffractive lenses and simulate the hyperspectral image formation.
- **Camera Modeling:** Convert spectral data to RGB using sensor response curves.
- **Network Reconstruction:** Train neural networks to reconstruct the original hyperspectral cube from the encoded RGB image.

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/singer-yang/DeepLens_HSI.git
cd DeepLens_HSI

# Create and activate the conda environment
conda env create -f environment.yml -n deeplens
conda activate deeplens
```

### Quick Start

```bash
# Warm up
python 0_hello_deeplens_hsi.py

# End-to-end hyperspectral image reconstrcution from encoded RGB images
python 1_deeplens_hsi.py
```

## Project Structure

- `deeplens/` - Core library
  - `hsi_camera.py` - Hyperspectral camera simulation
  - `diffraclens.py` - Diffractive lens implementation
  - `optics/` - Optical simulation components
  - `network/` - Neural network models for reconstruction
  - `utils/` - Utility functions
- `lenses/` - Lens files
- `sensors/` - Sensor files
- `configs/` - Training and evaluation configurations
- `datasets/` - Hyperspectral image datasets
- `1_deeplens_hsi.py` - Single-GPU training script


## License

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC).

- The license is only for non-commercial use (commercial licenses can be obtained from the authors)
- The material is provided as-is, with no warranties whatsoever
- If you publish any code, data, or scientific work based on this, please cite our work

## Citation

This repository is an application example built upon the DeepLens framework. If you use this code or concepts in your research, please cite the paper developing the **DeepLens framework**:

```bibtex
@inproceedings{Yang_2024,
  title={End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model},
  author={Yang, Xinge and Souza, Matheus and Wang, Kunyi and Chakravarthula, Praneeth and Fu, Qiang and Heidrich, Wolfgang},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  series={SA '24},
  pages={1--11},
  year={2024},
  month=dec,
  publisher={ACM},
  DOI={10.1145/3680528.3687640},
  url={http://dx.doi.org/10.1145/3680528.3687640}
}
```


## Ackonledgement

The height map of the DOE was provided by Jingyue Ma ("https://github.com/Jingyue-MA").
