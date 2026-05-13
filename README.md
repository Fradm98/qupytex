# `qupytex`

Quantum Phase Transition Explorer Python is a package to navigate researchers in the characterization of quantum phase transitions of Quantum Many-Body systems.

## Installation

To install use pip:
```
pip install qupytex
```

and provide also the useful dependencies by executing from terminal the following command:
```
pip install -r requirements.txt
```

## Package description and Usage

This library allows to prepare ground states of some quantum many-body Hamiltonian using dmrg algorithm, and to identify thanks to our novel approach, the phase transition of the model and its nature, characterizing its phase diagram.

Our method relies on the concept of reduced fidelity susceptibility (RFS). Taking a parameter of the phase space $\lambda$ and moving in one of the allowed directions by a $\delta \lambda$, we can compare the ground states relative to these two positions in the phase space. Naively, if the RFS does not change fast, it means we are in the same phase. A sudden phase in this quantity can mark the presence of a Quantum Phase Transition (QPT). 
Other than finding where the phase transition occurs, we can also study the nature of this transition, e.g. if it is of the second order, infinite order, and/or topological. This is done by the Order Parameter Discovery (OPD). We exploit the information obtained from the RFS to get an observable which, e.g decomposing in pauli matrices has different contributions. Focusing on a phase boundary, we claim that the main contributions of the decomposition of the found observable coincide with the order parameter of the theory, even when the latter is non-local.

The module `exec_dmrg.py` allows you to select the model to study. As aforementioned, the method is valid only for small deltas in the phase space. Thus, we select also the parameter extent and how dense we want our grid (plus all the typical dmrg params as length of the chain, physical space, and bond dimension).
Once we have our ground states, we use the module `exec_phase_diagram_detection.py` to load the ground states, compute the k-site reduced density matrices (depending on the model it will show or not show more features of the phase diagram) and compute the RFS. Taking the gradient of this give us a powerful visualization tool, that is, a vector field. The angles of this vector field are also related directly to the parameters of the Hamiltonian.
Eventually, we can use the module `exec_order_parameter_discovery.py` to go a step further.
(TODO generate a exec_OPD.py file to show a simple example)

## Acknowledgments
This repository is based on the work https://arxiv.org/abs/2408.01400v4 and we thank all the authors:
Nicola Mariella, Tara Murphy, Francesco Di Marcantonio, Khadijeh Najafi, Sofia Vallecorsa, Sergiy Zhuk, and Enrique Rico.