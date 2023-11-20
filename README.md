# Geothermal Ensemble Methods

This repository contains code for running ensemble Kalman methods for approximate Bayesian inference on a set of simple geothermal models.

## TODO:
### Slice
 - Check whether things are still working for this model after the repository refactor.
 - Use Matern fields, hyperparameters? Maybe just vary the lengthscale, not the standard deviations.
 - Whiten everything properly.
 - Update level set function to Gaussian copula-ish ideas?
 - Think about inverse crimes (i.e. make a finder model.)
 - Think about how the noise is generated.
 - Plot the data, truth, convective plume, vapour saturations?
 - Save data, truth, true states to file
### Channel
 - Add feedzone locations.
 - Tune Robin parameter on Matern fields.
 - Figure out what the level width should be.
 - Set up mechanism to save true parameters, states and data to file.
 - Develop method for visualising the convective plume of the model.
 - Think about inverse crimes (i.e. consider making a finer model)
 - Check everything carefully!
### Both Models
 - Think about the initial conditions more carefully (thermal gradient?)
### Ensemble Methods
 - Tidy up EKI-DMC and EnRML code (less object-oriented?)
### Misc
 - Update NZGW code and slides.
 - Finish writing README (add some notes on the various algorithms, other implementations, etc).
 - Make a Python environment for this repository?