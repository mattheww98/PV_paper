# The carbon cost of materials discovery: Can machine learning really accelerate the discovery of new photovoltaics?
This is a repo for the paper, pre-print [here](https://arxiv.org/abs/2507.13246).
## Datasets used
The datasets used in this paper are as follows:
- ~10k GGA & HSE band-gaps from Kim et al. (used to calculate GGA-HSE band-gap shifts to scissor correct GGA absorption spectra)
- ~18k GGA absorption spectra from Woods-Robinson et al. (filtered to exclude lanthanides and actinides in this work, used to training spectral models and to calculate SLMEs)
- ~700 GGA + Delta-sol corrected absorption spectra from Fabini et al. (used as test set)
- ~5k TBmBJ SLMEs from Choudhary et al. (used for comparison between functionals)

## Directories in this repo
The making_datasets directory has notebooks with examples for querying these datasets.
The remaining directories are as follows:
-  VASP_calculations - scripts to run ATOMATE2 to generate GGA & HSE VASP calculations and to wrap VASP calculations in codecarbon to track energy usage.
- GW_comparison - a comparison of the data used in this work to higher fidelity GW calculations from Yu and Zunger (2012), and ML predictions on their data using the ML models trained in this work.
- learning_curves - generation of subsets of the training data and post-processing of the predictions by models trained on theses subsets to produce learning curves
- predicting_properties - scripts and input files for training ALIGNN models to predict each of the properties (band-gaps, absorption spectra, etc.) considered in this work and post-processing scripts to produce the plots in the paper. The training/inference scripts have codecarbon wrappers for monitoring energy usage over CPUs and GPUs.

Each of these directories contains a README.txt for further information.
