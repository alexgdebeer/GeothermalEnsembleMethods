# Ensemble Methods for Geothermal Reservoir Modelling

![Fault Model](fault_model.png)

This repository contains code for running ensemble Kalman inversion (EKI; [[1](#1), [2](#2)]) and ensemble randomised maximum likelihood (EnRML; [[3](#3)]) for approximate Bayesian inference of the subsurface permeability structure and hot mass upflow of two simple geothermal reservoir models.
EKI and EnRML are derivative-free algorithms that iteratively update a set of interacting particles to produce an approximation of the posterior. 
They differ in terms of the dynamic under which the particles evolve.

This repository contains a two-dimensional vertical slice model and a three-dimensional model with a vertical fault running through the centre of the reservoir. 
The image at the top of this file shows the model mesh and well locations *(left)*, permeability structure *(centre)*, and natural state convective plume *(right)* of the three-dimensional model.
Both models are run using the [Waiwera](https://waiwera.github.io/) geothermal simulator.

## Getting Started

To run the code in this repository on a local machine, first install Python >= 3.8. Then, clone the repository and install the project dependencies by running the following at the command line:

```
python3.8 -m pip install -r requirements.txt
```

You will also need to install Waiwera. For information on getting started with Waiwera, consult the Waiwera [website](https://waiwera.github.io/install/).

To run EKI and EnRML on each model, use the scripts in the top level of the repository (adjusting the algorithm parameters where appropriate).

## References

[<a id="1">1</a>]
Iglesias, MA, Law, KJ, and Stuart, AM (2013).
Ensemble Kalman methods for inverse problems.
*Inverse Problems* **29**, 045001.

[<a id="2">2</a>]
Iglesias, M and Yang, Y (2021). 
Adaptive regularisation for ensemble Kalman inversion.
*Inverse Problems* **37**, 025008.

[<a id="3">3</a>]
Chen, Y and Oliver, DS (2013). 
Levenberg–Marquardt forms of the iterative ensemble smoother for efficient history matching and uncertainty quantification.
*Computational Geosciences* **17**, 689–703.