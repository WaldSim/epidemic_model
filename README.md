# Content

This repository contains python implementations of some epidemic models, notably realizations of the SIR and SIS models (see https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology).

**2D simulation on grid**

The file ```sir_ani.py``` contains a pure numpy implementation of the SIR model on a 2D, regular grid. Running it via
  ```$ python sir_ani.py```
opens an animation where the population dynamics on the grid is evaluated in real time (see the gif below).
Model parameters and grid size can by adjusted by setting the appropriate flag, e.g. type ```$ python sir_ani.py --gamma=2.0``` to set gamma to 0.2.

To list all possible options, just run ```$ python sir_ani.py -h```.

![](https://github.com/WaldSim/epidemic_model/blob/master/grid.gif)

**Model dynamics**

The notebook ```SIR.ipynb``` solves the SIR and SIS model with an efficient ordinary differential equation numerical solver. The notebook comes with a handy animation method, allowing the user to manually adjust the models' parameters and immediately see their respective effect on the dynamics of the epidemy.

![](https://github.com/WaldSim/epidemic_model/blob/master/sir.png)

## requirements

Nothing special here, the 2D solver only relies on numpy for the calculations and matplotlib for plotting. The notebook additionally needs an installation of scipy. All requirements can be found in the requirements.txt
