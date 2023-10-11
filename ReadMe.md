## Description

This repository contains the processing code used in the experiments of the paper "Heteroscedastic Gaussian Processes and
Random Features: Scalable Motion Primitives with Guarantees" (Edoardo Caldarelli, Antoine Chatalic, Adrià Colomé, Lorenzo Rosasco, Carme Torras), published in the 7th Conference on Robot Learning (CoRL), Atlanta, GA, USA, 2023.

## Sources
The data of the trajectories are taken from [https://codeocean.com/capsule/9688212/tree/v2]. The implementation of the heteroscedastic Gaussian process (HGP) with random features (RFs) is based on the gpflow Python package [https://www.gpflow.org/].

## Installation
The required packages can be installed by running the following command:

```
pip install -r requirements.txt
```

## Offline processing of trajectories

In order to process the trajectories of the three experiments offline, run the script ``process_trajs.py``. In this case, all the human demonstrations are merged and form a unique dataset, for each task. The results can be found in the experiments' directories, e.g., ```proof-of-concept```.

## Online processing of trajectories

In order to process the trajectories online, run the script ``process_trajs_online.py``. In this case, the posterior distribution of the HGP is subject to low-rank updates as soon as a new trajectory is processed. The experiments are run with RFs and with the Nyström method. Note that the exact GP needs to be fitted to the data via ``process_trajs.py`` prior to running this script.

## Utilitites 

The script ``MLHGP_RFFs.py`` contains the implementation of the EM algorithm used for training the RF-HGP and the exact HGP. The script ``MLHGP_variational.py`` contains the EM algorithm adapted to the sparse variationl Gaussian process (SVGP) by Hensman et al. (2013).
The script ``gp_utilities.py`` contains the implementation of the RF-based kernel, as well as the Nyström approximation used in our online processing experiments.

## Plotting code

This repository further contains the code that can be used to generate our plots. This code can be found in ``plot_metrics_with_rate.py`` (plotting the oracle experiments and the theoretical rates), 
``plot_metrics_online.py``, plotting the results of the incremental learning experiments, and ``plot_metrics_w_variational.py``, plotting the results of our heuristic experiments.
