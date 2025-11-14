# The carbon cost of materials discovery: Can machine learning really accelerate the discovery of new photovoltaics?
This is a repo for the paper, pre-print [here](https://arxiv.org/abs/2507.13246).
## Datasets used
The datasets used in this paper are as follows:
- ~10k GGA & HSE band-gaps from [Kim et al.](https://www.nature.com/articles/s41597-020-00723-8) (used to calculate GGA-HSE band-gap shifts to scissor correct GGA absorption spectra)
- ~18k GGA absorption spectra from [Woods-Robinson et al.](https://www.sciencedirect.com/science/article/pii/S2590238523003545#abs0015) (filtered to exclude lanthanides and actinides in this work, used to training spectral models and to calculate SLMEs)
- ~700 GGA + Delta-sol corrected absorption spectra from [Fabini et al.](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.8b04542) (used as test set)
- ~5k TBmBJ SLMEs from [Choudhary et al.](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.9b02166) (used for comparison between functionals)

## Directories in this repo
The making_datasets directory has notebooks with examples for querying these datasets.
The remaining directories are as follows:
- VASP_calculations - scripts to run [ATOMATE2](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00019j) to generate GGA & HSE VASP calculations and to wrap VASP calculations in [codecarbon](https://codecarbon.io/) to track energy usage.
- GW_comparison - a comparison of the data used in this work to higher fidelity GW calculations from [Yu and Zunger (2012)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.068701), and ML predictions on their data using the ML models trained in this work.
- learning_curves - generation of subsets of the training data and post-processing of the predictions by models trained on theses subsets to produce learning curves
- predicting_properties - scripts and input files for training ALIGNN models to predict each of the properties (band-gaps, absorption spectra, etc.) considered in this work and post-processing scripts to produce the plots in the paper. The training/inference scripts have codecarbon wrappers for monitoring energy usage over CPUs and GPUs.

Each of these directories contains a README.txt for further information.
