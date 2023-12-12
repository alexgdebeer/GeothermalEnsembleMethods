# Geothermal Ensemble Methods

This repository contains code for running ensemble Kalman methods for approximate Bayesian inference on a set of simple geothermal models.

## Running Locally

To run the code in this repository on a local machine, first install Python >= 3.8. 

Then, install the project dependencies by running the following at the command line:
```
python3.{VERSION} -m pip install -r requirements.txt
```
where ```{VERSION}``` is replaced with your version of Python.

## TODO:
### Slice
 - N/A
### Channel
 - Generate and visualise data.
 - Tune Robin parameter on Matern fields.
 - Think about inverse crimes (i.e. consider making a finer model)
### Both Models
 - Check everything carefully!
### Ensemble Methods
 - Fix EnRML code.
### Misc
 - Update NZGW code and slides.
 - Finish writing README (add some notes on the various algorithms, other implementations, etc).