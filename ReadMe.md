<p align="center">
   <img src="https://github.com/Emmanuelpean/pears/blob/main/resources/medias/logo_text.svg" alt="Pears">
</p>
<p align="center">
   <a href="https://opensource.org/licenses/MIT">
   <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Licence">
   </a>
   <a href="https://github.com/Emmanuelpean/pears/actions?query=branch%3Amain+event%3Apush">
   <img src="https://github.com/emmanuelpean/pears/actions/workflows/test.yml/badge.svg?event=push&branch=main" alt="Testing">
   </a>
   <a>
   <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/emmanuelpean/3d13cd09334063855921d2537ee75916/raw/pytest-coverage-comment__main.json" alt="Coverage">
   </a>
   <a href="https://github.com/psf/black">
   <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
   </a>
</p>

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

App created and maintained by Emmanuel V. Pean.  
Version 2.0.0 (last updated: March 2025).  
If you use *Pears* to fit your TRPL or TRMC data, please cite https://doi.org/10.1021/acs.jcim.3c00217.
