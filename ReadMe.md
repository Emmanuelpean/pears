<p align="center">
  <img src="https://github.com/Emmanuelpean/pears/blob/main/resources/medias/logo.svg" alt="Raft" width="150">
</p>

<h1 align="center">PEARS</h1>
<h2 align="center">Perovskite Recombination Simulator</h2>

<div align="center">

  [![Passing](https://github.com/emmanuelpean/pears/actions/workflows/test.yml/badge.svg?branch=main&event=push)](https://github.com/Emmanuelpean/pears/actions/workflows/test.yml)
  [![Tests Status](./reports/tests/tests-badge.svg?dummy=8484744)](https://emmanuelpean.github.io/pears/reports/tests/report.html?sort=result)
  [![Coverage Status](./reports/coverage/coverage-badge.svg?dummy=8484744)](https://emmanuelpean.github.io/pears/reports/coverage/htmlcov/index.html)
  [![Last Commit](https://img.shields.io/github/last-commit/emmanuelpean/pears/main)](https://github.com/emmanuelpean/pears/commits/main)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>


*Pears* is a user-friendly web app designed to fit time-resolved photoluminescence (TRPL) and time-resolved microwave 
photoconductivity (TRMC) data of perovskite materials using state-of-the-art charge carrier recombination models 
(extensively discussed for TRPL in https://doi.org/10.1039/D0CP04950F): 
* The **Bimolecular-Trapping-Auger** (BTA) accounts for bimolecular band-to-band recombination, monomolecular trapping and 
trimolecular Auger recombination. Because Auger recombination are usually non-negligible only at very high excitation 
fluences, Auger recombination are usually ignored within this model (the Auger recombination rate constant $k_A$ is 
fixed to 0 by default). This model assumes that the trap states remain mostly empty and that doping is negligible. 
As a result, this model is relatively simple and less likely to lead to over-parameterisation.
* The **Bimolecular-Trapping-Detrapping** (BTD) model accounts for bimolecular band-to-band recombination, bimolecular 
trapping, and bimolecular detrapping. Contrary to the BTA model, the BTD model accounts for the population and 
depopulation of the trap states, as well as the presence of doping. However, the increased complexity of this model can 
lead to over-parameterisation and ambiguous results.

*Pears* offers two operational modes:
* The **Fitting** mode is the primary mode for fitting experimental TRPL and TRMC data.
* The **Grid Fitting** mode allows to run the fitting optimisation across a range of guess parameters, helping to 
identify whether multiple sets of parameters yield fitting solutions, as discussed in https://doi.org/10.1002/smtd.202400818. 
If the optimisations do not converge to the same values, the fitting may be inaccurate due to the presence of multiple 
possible solutions.

If you use *Pears* to fit your TRPL or TRMC data, please cite https://doi.org/10.1021/acs.jcim.3c00217.

## Installation

Create and activate a virtual environment, activate it, and run:
```console
$ pip install -e .[dev]
```

## Usage
To run the app locally, run:
```console
$ cd app
$ streamlit run ./main.py
```
