## Overview

Welcome! This repository contains the data and code needed to reproduce the results from the paper, titled "Biased belief priors versus biased belief updating: differential correlates of depression and anxiety", authored by Christopher Gagne, Sharon Agai, Christian Ramiro, Peter Dayan & Sonia J. Bishop, and published in PLOS Computational Biology.

This Github repository will also be stored within an Open Science Framework (OSF) repository, where it will be hosted permanently: https://osf.io/zr7ne/

## Steps for Reproducing the Main Analyses

1. Download the Github repository from the OSF repository
2. Install the required Python environments
3. Fit the behavioral models to data (optional step; fitted models already contained in repo)
4. Plot main and supplemental figures from the results

## (1) Download Github Repository

You should download the entire Github repository from the OSF repository (rather than downloading individual folders). There are some code dependencies between folders.

## (2) Installations

The easiest way to reproduce the code is to install a conda environment, using the following bash command:

```
conda env create -f environment.yml
```

To activate the environments use `conda activate env_belief_updating`.

## (3) Fitting Behavioral Models to Data

We recruited participants in three separate rounds. Each participant completed up to three experimental sessions. Data for participants can be found in `/data/` in folders labeled according to the recruitment round (i.e., 'round1','round2','round3') and experiment session (i.e. 'session1', 'session2', 'session3'); see Paper for more details about data collection. For a more detailed description of the data contained in each folder and file, see `/data/data_description.md`.

The code used to load these data files and prepare them for either model fitting or figure creation can be found in the folder `/code_for_data_processing`. However, this is supporting code and does not need to be called directly.

Fitting behavioral models to participants data can be done using code found in the `/code_for_modeling_fitting` folder.

The main model (\#3) can be fit to experiment data by running the following bash command:

```
python fit_one_model.py --seed 1 --modelname model_RW_update_w_report_samelr_relfbscaling --subjblock all --prior True --save True
```

Note that this code needs to be run in the Python conda environment. It depends on additional supporting code found in the `/models.py` module.

## (4) Plotting Main and Supplemental Figures

The code used to generate Figures 2-5 can be found in the folder `/notebooks`. The code is contained in the Jupyter notebooks `Main_Figures.ipynb`.

The code used to generate the supplemental figures (S1-S9) can also be found in the folder `/notebooks`, and is contained in the Jupyter notebooks `Supplemental_Figures.ipynb`.
