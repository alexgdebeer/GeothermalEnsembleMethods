# Geothermal Ensemble Methods

This repository contains code for running ensemble Kalman methods for approximate Bayesian inference on a set of simple geothermal models.

## TODO:
### Slice
 - Think about inverse crimes (i.e. make a finder model)
 - Think about how the noise is generated (increase to 5-10%)
 - Plot the data, truth, convective plume, vapour saturations?
### Channel
 - Add feedzone locations.
 - Tune Robin parameter on Matern fields.
 - Figure out what the level width should be.
 - Set up mechanism to save true parameters, states and data to file.
 - Develop method for visualising the convective plume of the model.
 - Think about inverse crimes (i.e. consider making a finer model)
 - Save data, truth, true states to file
### Both Models
 - Read paper on how to adjust ensemble covariance when defining Gaussian based on successful ensemble members.
 - Think about the initial conditions more carefully (thermal gradient?)
 - Check everything carefully!
### Ensemble Methods
 - Tidy up EKI-DMC and EnRML code (less object-oriented?)
### Misc
 - Update NZGW code and slides.
 - Finish writing README (add some notes on the various algorithms, other implementations, etc).
 - Make a Python environment for this repository?