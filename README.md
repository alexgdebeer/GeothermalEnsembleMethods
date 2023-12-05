# Geothermal Ensemble Methods

This repository contains code for running ensemble Kalman methods for approximate Bayesian inference on a set of simple geothermal models.

## Getting Started

To run the code in this repository, first ensure that Python 3.8 is installed on your computer. Then, create a virtual environment with the project dependencies by running the following at the command line:

```
python3.8 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

## TODO:
### Slice
 - Think about how the noise is generated (increase to 5-10%?)
 - Plot the vapour saturations for the true system (could be interesting)
 - Update prior to be able to extract hyperparameters
### Channel
 - Add feedzone locations.
 - Tune Robin parameter on Matern fields.
 - Figure out what the level width should be.
 - Set up mechanism to save true parameters, states and data to file.
 - Develop method for visualising the convective plume of the model.
 - Think about inverse crimes (i.e. consider making a finer model)
 - Save data, truth, true states to file
 - Rescale the pressures and enthalpies appropriately
### Both Models
 - Read paper on how to adjust ensemble covariance when defining Gaussian based on successful ensemble members.
 - Think about the initial conditions more carefully (thermal gradient?)
 - Check everything carefully!
 - Add an Ensemble object into models.py
### Ensemble Methods
 - Tidy up EnRML code
### Misc
 - Update NZGW code and slides.
 - Finish writing README (add some notes on the various algorithms, other implementations, etc).
 - Make a Python environment for this repository?