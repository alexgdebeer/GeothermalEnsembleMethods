# Geothermal Ensemble Methods

This repository contains code for running ensemble Kalman methods for approximate Bayesian inference on a set of simple geothermal models.

## Running Locally

To run the code in this repository on a local machine, first install Python >= 3.8. Then, create a virtual environment with the project dependencies by running the following at the command line:
```
python3.{VERSION} -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
where ```{VERSION}``` is replaced with your version of Python.

## TODO:
### Slice
 - N/A
### Channel
 - Generate and visualise data.
 - Tune Robin parameter on Matern fields.
 - Figure out what the level width should be.
 - Set up mechanism to save true parameters, states and data to file.
 - Develop method for visualising the convective plume of the model.
 - Think about inverse crimes (i.e. consider making a finer model)
### Both Models
 - Read paper on how to adjust ensemble covariance when defining Gaussian based on successful ensemble members.
 - Check everything carefully!
### Ensemble Methods
 - Fix EnRML code.
### Misc
 - Update NZGW code and slides.
 - Finish writing README (add some notes on the various algorithms, other implementations, etc).