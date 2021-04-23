# pyPplusS

pyPplusS is a code package accompanying the paper "Fast and precise light-curve model for transiting exoplanets with rings" ([doi: 10.1093/mnras/stz2556](https://doi.org/10.1093/mnras/stz2556), ads bibcode: 2019MNRAS.490.1111R).
pyPplusS is a model for the light curve of ringed exoplanets transits’ for uniform and limb darkened stars. The calculation is done using the Polygon+Segments algorithm, described in the above paper.

pyPplusS provides calculation of light curves for ringed, oblate or spherical exoplanets in both the uniform and limb darkened cases.

## Installation
Run the following to install pyPplusS:
```pip install pyppluss```


## Package Structure

The package is split into five files:
- polygon_plus_segments is an implementation of the Polygon+Segments algorithm described in the above paper.
- segment_models contains functions for modelling light curves in the uniform and limb darkened cases.
- err_order_fin is a script described in appendix C, which aids in setting the order of numerical integration.
- base_functions contains a variety of "low-level" helpers.
- fastqs is a python implementation of the Fast Quartic Solver described by [Strobach (2010)](https://www.sciencedirect.com/science/article/pii/S0377042710002128).

Detailed descriptions of all inputs and outputs are given by comment lines and documentation text in the code.

## Usage
To use pyPplusS, import the following function: ```from pyppluss.segment_models import LC_ringed```

This function will return the light curve of ringed exoplanets transits’ for limb darkened stars.

## Dependencies

Dependencies: numpy, scipy, python 3.

The package was tested with numpy 1.16.3, scipy 0.19.1, python 3.6.8.

## Issues
Issues can be reported at GitHub, at the EdanRein/pyPplusS repository.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments
This package was written by Edan Rein and Dr. Aviv Ofir, during work at Prof. Oded Aharonson's lab at the 
Center for Planetary Science in the Department of Earth and Planetary Sciences, Weizmann Institute of Science.

## Citations
If you use this code, please cite [Rein and Ofir (2019)](https://doi.org/10.1093/mnras/stz2556).
The abstract is available [here](https://academic.oup.com/mnras/article-abstract/490/1/1111/5568385), and
the paper is available on arXiv [here](https://arxiv.org/abs/1910.12099).
