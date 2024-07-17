# PULSEY

A python package for modeling stellar pulsation. Due to deprecations, a python environment must be created (with latest possible version=3.9.19) and necessary packages must be installed. For ease of the user, a list of terminal command lines to create the environment and install the required packages follows below:


conda create -c conda-forge -n pymc_env "pymc3" "libblas=*=*accelerate"

conda activate environment

pip install starry

pip install pulsey

pip install tqdm

pip install exoplanet_core

pip install ipympl
