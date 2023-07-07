# Equivariance-aware Architectural Optimization of Neural Networks

This repository contains code in the work "[Equivariance-aware Architectural Optimization of Neural Networks](https://openreview.net/forum?id=a6rCdfABJXg)" presented at ICLR 2023. We implement
* Two mechanisms towards equivariance-aware architectural optimization: the equivariance relaxation morphism and the $[G]$-mixed equivariant layer
* Two equivariance-aware neural architecture search (NAS) algorithms that implement these mechanisms respectively: an evolutionary method <nobr>EquiNAS<sub>$E$</sub></nobr> and a differentiable method <nobr>EquiNAS<sub>$D$</sub></nobr>
* Experiments in image classification with finite rotation and reflection symmetry groups

<nobr>EquiNAS<sub>$E$</sub></nobr> may be run via `hillclimb.py` and <nobr>EquiNAS<sub>$D$</sub></nobr> may be run via `deanas.py`.


Recommended citation:

Maile, Kaitlin, Dennis G. Wilson, and Patrick Forré. "Equivariance-aware Architectural Optimization of Neural Networks." In The Eleventh International Conference on Learning Representations. 2023.